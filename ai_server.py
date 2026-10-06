import os

from dotenv import load_dotenv

# .env를 Google/gRPC 라이브러리 import 전에 읽어 DNS resolver 설정이
# 실시간 STT 채널 생성에도 확실히 적용되도록 한다.
load_dotenv()
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

import math
import time
import json
import random
from datetime import datetime
from threading import Lock, Thread
from queue import Queue

import pandas as pd
import numpy as np
import requests

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sock import Sock

from groq import Groq
import google.auth
from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions

from pydantic import BaseModel, ConfigDict
from typing import List, Optional

# MILP(Model B)용. requirements.txt에 pulp가 없으면 자동으로
# 그리디(반복 최적화) 방식으로 대체 동작한다. (아래 rank_model_b 참고)
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False


# =========================================================
# 1. 환경변수 및 Flask 설정
# =========================================================

# 환자 상태 구조화용 LLM은 Groq API의 GPT-OSS 120B를 사용한다.
# 무료 API 사용량 보호를 위해 SDK 자동 재시도는 사용하지 않는다.
GROQ_API_KEY = str(os.getenv("GROQ_API_KEY") or "").strip()
GROQ_MODEL = str(os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b").strip()

# Google Cloud Speech-to-Text V2 설정.
# GOOGLE_CLOUD_PROJECT가 .env에 없으면 gcloud ADC에서 프로젝트 ID를 자동 탐지한다.
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_STT_LOCATION = os.getenv("GOOGLE_STT_LOCATION", "global")
GOOGLE_STT_MODEL = os.getenv("GOOGLE_STT_MODEL", "short")
GOOGLE_STREAMING_STT_LOCATION = os.getenv("GOOGLE_STREAMING_STT_LOCATION", "us")
GOOGLE_STREAMING_STT_MODEL = os.getenv("GOOGLE_STREAMING_STT_MODEL", "chirp_3")
GOOGLE_STT_MAX_AUDIO_BYTES = 10 * 1024 * 1024
GOOGLE_STT_MAX_RECORDING_SEC = 30

# 응급의료 도메인 용어. Cloud Speech-to-Text model adaptation의 inline PhraseSet에만 사용한다.
# 병원 추천/LLM 환자 분석 로직에는 영향을 주지 않는다.
GOOGLE_STT_MEDICAL_PHRASES = [
    "말이 어눌하다",
    "말이 어눌해지고",
    "말이 어눌해졌습니다",
    "오른쪽 팔",
    "오른쪽 다리",
    "왼쪽 팔",
    "왼쪽 다리",
    "팔과 다리에 힘이 빠졌습니다",
    "편마비",
    "의식저하",
    "의식 소실",
    "뇌졸중",
    "뇌출혈",
    "CT 검사",
    "MRI 검사",
    "심근경색",
    "급성 심근경색",
    "심전도",
    "ST 상승",
    "흉통",
    "심혈관 중재",
    "호흡곤란",
    "산소포화도",
    "SpO2",
    "수축기혈압",
    "이완기혈압",
    "혈압",
    "맥박",
    "호흡수",
    "체온",
    "저혈압",
    "고열",
    "구토",
    "설사",
    "소변량이 줄었습니다",
    "중증 탈수",
    "소아 응급 진료",
    "수액 치료",
    "경련",
    "심한 두통",
    "전자간증",
    "임신중독증",
    "고위험 산모",
    "임신 34주",
    "다발성 외상",
    "교통사고 후",
    "복부 통증",
    "골절",
    "소아 탈수",
    "중환자실",
    "중환자 치료",
    "ICU",
    "응급 수술",
    "혈관조영",
    "응급의학과",
    "신경과",
    "신경외과",
    "심장혈관흉부외과",
    "산부인과",
    "소아청소년과",
]

# 나이/성별은 응급상황에서 핵심 식별 정보이므로 별도 adaptation phrase로 보강한다.
# 실제 음성에 없는 나이를 후처리로 추정하지 않고, STT 단계에서 "N세 + 성별" 발음을
# 더 안정적으로 인식하도록 bias만 준다.
def _build_google_stt_demographic_phrases():
    phrases = []

    # 일반 연령 표현: "68세", "68세 남성", "68세 여성"
    for age in range(1, 101):
        phrases.extend([
            f"{age}세",
            f"{age}세 남성",
            f"{age}세 여성",
        ])

    # 소아에서 자주 쓰는 표현.
    for age in range(1, 19):
        phrases.extend([
            f"{age}세 남아",
            f"{age}세 여아",
        ])

    # 영아 월령 표현.
    for month in range(1, 25):
        phrases.extend([
            f"생후 {month}개월",
            f"생후 {month}개월 남아",
            f"생후 {month}개월 여아",
        ])

    # 순서를 유지하면서 중복 제거.
    return list(dict.fromkeys(phrases))


GOOGLE_STT_DEMOGRAPHIC_PHRASES = _build_google_stt_demographic_phrases()


def _google_stt_adaptation_phrase_entries():
    """Cloud STT inline PhraseSet에 넣을 phrase/boost 목록."""

    entries = [
        {"value": phrase, "boost": 10.0}
        for phrase in GOOGLE_STT_MEDICAL_PHRASES
    ]

    entries.extend(
        {"value": phrase, "boost": 12.0}
        for phrase in GOOGLE_STT_DEMOGRAPHIC_PHRASES
    )

    return entries


def _google_stt_adaptation_phrase_count():
    return (
        len(GOOGLE_STT_MEDICAL_PHRASES)
        + len(GOOGLE_STT_DEMOGRAPHIC_PHRASES)
    )

if not GROQ_API_KEY:
    raise ValueError(
        "Groq API 키가 .env에 없습니다. GROQ_API_KEY를 설정해 주세요."
    )

# Groq Python SDK는 일부 오류를 기본적으로 자동 재시도하므로 명시적으로 끈다.
# 동일 환자 입력은 아래 PatientInfo 캐시를 통해 재사용한다.
GROQ_CLIENT = Groq(
    api_key=GROQ_API_KEY,
    max_retries=0,
)

print(
    f"[GROQ] 환자 상태 구조화 모델={GROQ_MODEL}, "
    "reasoning_effort=medium, strict structured output=ON, SDK 자동 재시도=OFF"
)

app = Flask(__name__, static_folder=".")
CORS(app)
sock = Sock(app)

# Streamlit에서 Excel 저장 없이 넘겨받는 최신 병원 DB.
# Flask가 재시작되면 사라지고, 그때는 아래 기존 Excel fallback 로직을 그대로 사용한다.
RUNTIME_HOSPITAL_DF = None
RUNTIME_HOSPITAL_META = {"source": None, "synced_at": None, "db_updated_at": None}
RUNTIME_DB_LOCK = Lock()

# 프론트에서 GPS 위치를 못 받아왔을 때 사용하는 기본 위치(대전광역시청).
# 실제 서비스에서는 반드시 구급차의 실시간 GPS 좌표를 프론트에서 전달해야 한다.
DEFAULT_AMBULANCE_LAT = 36.3504
DEFAULT_AMBULANCE_LNG = 127.3845

# 카카오모빌리티 REST API 키. 기존 지도팀의 server.js 역할을 Flask 백엔드 안으로 합친다.
# .env에 KAKAO_REST_API_KEY가 있어야 실제 도로 경로/실시간 교통 ETA를 조회할 수 있다.
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")
KAKAO_JAVASCRIPT_KEY = os.getenv("KAKAO_JAVASCRIPT_KEY")
# 병원 DB 관리 웹(Streamlit) 주소. Render에서는 환경변수로 실제 배포 주소를 지정한다.
DB_MANAGER_URL = str(os.getenv("DB_MANAGER_URL") or "http://localhost:8501").strip().rstrip("/")
KAKAO_REQUEST_TIMEOUT_SEC = 15

# 환자 위치 랜덤 생성 범위. 지도팀 코드의 전국 시연용 반경과 같은 의미다.
PATIENT_MIN_RADIUS_KM = 0.8
PATIENT_MAX_RADIUS_KM = 5.0

# 병원 사전 수용 신청/이송 시작 시 병원측에 전송한 환자 정보를 기록하는 로그 파일.
# 실제 병원 시스템 연동 API가 준비되기 전까지는 이 파일이 "전송 기록"의 역할을 한다.
DISPATCH_LOG_FILE = "dispatch_log.jsonl"


# 개선 ver_0920의 Model B/C 계산에 필요한 보수적 프록시/설정.
# 실제 거절·재이송 이력이 쌓이기 전까지 사용하는 추정치이며 학습된 확률이 아니다.
ASSUMED_REJECTION_RESEARCH_MIN = 20.0
ASSUMED_RETRANSFER_MIN = 30.0
MIN_EXPECTED_WAIT_MIN = 5.0
CONGESTION_WAIT_RANGE_MIN = 25.0

# 카카오 다중 목적지 길찾기 API는 한 요청당 최대 30개 목적지를 받는다.
# 1차 batch에서 카카오 경로가 10개 미만일 때만 2차 batch를 추가로 조회한다.
# 따라서 추천 요청당 외부 다중 목적지 호출은 최대 2회로 제한한다.
KAKAO_RANKING_CANDIDATE_LIMIT = 30
KAKAO_MAX_BATCHES = 2
KAKAO_TARGET_CONFIRMED_ROUTES = 10

# 동일한 환자 위치에서 A/B/C/D를 비교할 때는 같은 Kakao 거리/ETA를 재사용한다.
# 다중 목적지 API에서 경로를 받지 못한 후보도 '수용 불가'로 간주하지 않는다.
# 해당 실패 상태 역시 TTL 동안 캐시하여 모델 전환 때 같은 실패 후보를 반복 호출하지 않고,
# 그 병원은 아래의 직선거리 기반 보수적 추정값으로 계속 후보에 남겨 둔다.
KAKAO_ROUTE_CACHE_TTL_SEC = 15 * 60
KAKAO_ROUTE_CACHE = {}
KAKAO_ROUTE_CACHE_LOCK = Lock()

TOPSIS_VARIANTS = {
    "C0": {
        "label": "기존형(기준선)",
        "description": "v8의 원형 TOPSIS. 기존 결과를 비교하기 위한 대조군입니다.",
        "patient_aware": False,
        "experimental": False,
        "criteria": {
            "acceptance_score": {"benefit": True, "weight": 0.40},
            "distance_km": {"benefit": False, "weight": 0.30},
            "required_specialist_count_total": {"benefit": True, "weight": 0.20},
            "congestion_score": {"benefit": False, "weight": 0.10},
        },
    },
    "C1": {
        "label": "보수적 개선형",
        "description": "ETA, 혼잡도, 전문치료 여유도, 데이터 신뢰도를 고정 가중치로 평가합니다.",
        "patient_aware": False,
        "experimental": False,
        "criteria": {
            "eta_min": {"benefit": False, "weight": 0.35},
            "congestion_score": {"benefit": False, "weight": 0.20},
            "specialty_margin": {"benefit": True, "weight": 0.30},
            "data_reliability_score": {"benefit": True, "weight": 0.15},
        },
    },
    "C2": {
        "label": "환자 맞춤형",
        "description": "C1과 동일한 기준을 사용하되 KTAS/골든타임에 따라 가중치를 변경합니다.",
        "patient_aware": True,
        "experimental": False,
        "criteria_from": "C1",
        "weight_profiles": {
            "critical": {
                "eta_min": 0.40,
                "congestion_score": 0.15,
                "specialty_margin": 0.35,
                "data_reliability_score": 0.10,
            },
            "standard": {
                "eta_min": 0.30,
                "congestion_score": 0.25,
                "specialty_margin": 0.20,
                "data_reliability_score": 0.25,
            },
        },
    },
    "C3": {
        "label": "치료지연 프록시 실험형",
        "description": "예상 치료시작 지연 프록시와 전문치료 여유도만 사용하는 실험안입니다.",
        "patient_aware": False,
        "experimental": True,
        "criteria": {
            "effective_treatment_delay_min": {"benefit": False, "weight": 0.70},
            "specialty_margin": {"benefit": True, "weight": 0.30},
        },
    },
}
DEFAULT_TOPSIS_VARIANT = "C1"

# 같은 환자 문장을 A/B/C/D 비교에서 반복 분석하지 않기 위한 메모리 캐시.
# Flask 재시작 시 자동으로 비워지며, 환자 문장이 달라지면 새 Groq 분석을 수행한다.
PATIENT_ANALYSIS_CACHE = {}
PATIENT_ANALYSIS_CACHE_LOCK = Lock()
PATIENT_ANALYSIS_CACHE_MAX = 100

MODEL_INFO = {
    "A": {
        "name": "Model A",
        "label": "가장 가까운 병원 추천",
        "description": "Hard Filter 통과 후 카카오 실제 도로거리/ETA를 확보한 후보 중 도로거리가 가까운 순으로 정렬합니다.",
    },
    "B": {
        "name": "Model B",
        "label": "MILP 혼합정수선형계획",
        "description": "카카오 ETA를 포함한 예상 치료시작 지연, 임상 적합성, 정보 신뢰도와 골든타임 위반을 반영합니다.",
    },
    "C": {
        "name": "Model C",
        "label": "TOPSIS 다기준 의사결정",
        "description": "공통 TOPSIS 엔진으로 C0/C1/C2/C3 변형을 비교하며 기본값은 C1입니다. ETA/거리 기준은 카카오 경로값을 사용합니다.",
    },
    "D": {
        "name": "Model D",
        "label": "가중치 종합 점수",
        "description": "수용점수, 카카오 도로거리, 전문의 수, 병원 혼잡도를 정규화한 뒤 사전 정의 가중치로 합산합니다.",
    },
}



# =========================================================
# 2. 환자 정보 구조
# =========================================================

class PatientInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suspected_category: str
    ktas: int
    golden_time_min: int
    specialists: List[str]
    equipment: List[str]
    req_icu: bool
    req_or: bool
    req_er_bed: bool
    req_severe_acceptance: bool


# =========================================================
# 3. 병원 DB 읽기
# =========================================================

def get_hospital_excel_file():
    """
    SAVER 추천 시스템에서 사용할 병원 DB 파일 선택.
    Streamlit에서 [엑셀로 저장]을 누르면 생성되는 saver_current_hospital_db.xlsx를 최우선 사용한다.
    """

    candidates = [
        "saver_current_hospital_db.xlsx",
        "national_hospital_db_departments_hira_updated.xlsx",
        "national_hospital_db_realtime_updated.xlsx",
        "national_hospital_master.xlsx",
        "daejeon_hospital_db_realtime_updated.xlsx",
        "daejeon_hospital_db.xlsx",
    ]

    for file in candidates:
        if os.path.exists(file):
            return file

    return None


def choose_sheet_name(excel_file):
    xls = pd.ExcelFile(excel_file)

    preferred_sheets = [
        "national_hospital_db",
        "hospital_master_realtime",
        "hospital_db_realtime",
        "hospital_master",
        "national_hospital_master",
        "matched_only",
    ]

    for sheet in preferred_sheets:
        if sheet in xls.sheet_names:
            return sheet

    return xls.sheet_names[0]


def load_hospital_db():
    # 1순위: Streamlit이 방금 전달한 최신 메모리 DB
    global RUNTIME_HOSPITAL_DF

    with RUNTIME_DB_LOCK:
        if RUNTIME_HOSPITAL_DF is not None and not RUNTIME_HOSPITAL_DF.empty:
            df = RUNTIME_HOSPITAL_DF.copy()
            df = df.drop(
                columns=[
                    col
                    for col in df.columns
                    if str(col).startswith("Unnamed")
                ],
                errors="ignore",
            )
            df = df.fillna("")

            print(
                f"[DB LOAD] runtime_streamlit_db, rows={len(df)}, "
                f"synced_at={RUNTIME_HOSPITAL_META.get('synced_at')}"
            )

            return df, "runtime_streamlit_db", "runtime"

    # 2순위 이하: 기존 Excel fallback 로직을 그대로 사용
    excel_file = get_hospital_excel_file()

    if excel_file is None:
        raise FileNotFoundError(
            "병원 DB 파일이 없습니다. "
            "saver_current_hospital_db.xlsx 또는 병원 DB 파일이 필요합니다."
        )

    sheet_name = choose_sheet_name(excel_file)
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    df = df.drop(
        columns=[
            col
            for col in df.columns
            if str(col).startswith("Unnamed")
        ],
        errors="ignore",
    )

    df = df.fillna("")

    print(
        f"[DB LOAD] file={excel_file}, "
        f"sheet={sheet_name}, rows={len(df)}"
    )

    return df, excel_file, sheet_name


@app.route("/api/sync-hospitals", methods=["POST"])
def sync_hospitals_runtime_api():
    """
    Streamlit의 현재 hospital_df를 SAVER 메모리 DB로 직접 전달한다.
    Excel 파일은 수정하거나 저장하지 않는다.
    """
    global RUNTIME_HOSPITAL_DF, RUNTIME_HOSPITAL_META

    try:
        data = request.get_json(silent=True) or {}
        hospitals = data.get("hospitals")

        if not isinstance(hospitals, list) or len(hospitals) == 0:
            return jsonify({
                "success": False,
                "error": "hospitals 리스트가 비어 있습니다."
            }), 400

        df = pd.DataFrame(hospitals)
        df = df.drop(
            columns=[
                col
                for col in df.columns
                if str(col).startswith("Unnamed")
            ],
            errors="ignore",
        )

        if df.empty:
            return jsonify({
                "success": False,
                "error": "전달된 병원 데이터가 비어 있습니다."
            }), 400

        with RUNTIME_DB_LOCK:
            RUNTIME_HOSPITAL_DF = df.copy()
            RUNTIME_HOSPITAL_META = {
                "source": data.get("source") or "streamlit_runtime",
                "synced_at": (
                    data.get("synced_at")
                    or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ),
                "db_updated_at": data.get("db_updated_at"),
            }

        print(
            f"[RUNTIME DB SYNC] rows={len(df)}, "
            f"cols={len(df.columns)}, "
            f"source={RUNTIME_HOSPITAL_META['source']}"
        )

        return jsonify(
            {
                "success": True,
                "hospital_count": len(df),
                "column_count": len(df.columns),
                "source": RUNTIME_HOSPITAL_META["source"],
                "synced_at": RUNTIME_HOSPITAL_META["synced_at"],
                "db_updated_at": RUNTIME_HOSPITAL_META.get("db_updated_at"),
            }
        )

    except Exception as e:
        print("RUNTIME DB SYNC ERROR:", e)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500




@app.route("/api/runtime-db-status", methods=["GET"] )
def runtime_db_status_api():
    with RUNTIME_DB_LOCK:
        meta = dict(RUNTIME_HOSPITAL_META)
        hospital_count = 0 if RUNTIME_HOSPITAL_DF is None else len(RUNTIME_HOSPITAL_DF)
    return jsonify({
        "success": True,
        "hospital_count": hospital_count,
        "source": meta.get("source"),
        "synced_at": meta.get("synced_at"),
        "db_updated_at": meta.get("db_updated_at"),
    })

def to_json_safe(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, float) and (
        math.isnan(value)
        or math.isinf(value)
    ):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def dataframe_to_records(df):
    records = []

    for row in df.to_dict(orient="records"):
        clean_row = {}

        for key, value in row.items():
            clean_row[key] = to_json_safe(value)

        records.append(clean_row)

    return records


@app.route("/api/hospitals", methods=["GET"])
def get_hospitals():
    try:
        df, excel_file, sheet_name = load_hospital_db()
        hospitals = dataframe_to_records(df)

        return jsonify(
            {
                "success": True,
                "source_file": excel_file,
                "sheet_name": sheet_name,
                "hospital_count": len(hospitals),
                "hospitals": hospitals,
            }
        )

    except Exception as e:
        print("EXCEL READ ERROR:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# 3-1. Google Cloud Speech-to-Text 음성 전사
# =========================================================

def get_google_cloud_project_id():
    """
    Speech-to-Text V2 요청에 사용할 Google Cloud 프로젝트 ID를 반환한다.

    1순위:
    .env의 GOOGLE_CLOUD_PROJECT

    2순위:
    gcloud auth application-default login으로 만든 ADC의 프로젝트
    """

    if GOOGLE_CLOUD_PROJECT:
        return GOOGLE_CLOUD_PROJECT

    try:
        _, detected_project_id = google.auth.default()

    except Exception as e:
        raise RuntimeError(
            "Google Cloud 인증 정보를 찾지 못했습니다. "
            "터미널에서 "
            "'gcloud auth application-default login'을 실행해 주세요."
        ) from e

    if not detected_project_id:
        raise RuntimeError(
            "Google Cloud 프로젝트 ID를 찾지 못했습니다. "
            ".env에 GOOGLE_CLOUD_PROJECT=saver-stt 를 추가해 주세요."
        )

    return detected_project_id


def build_google_stt_config():
    """
    한국어 단문 음성 전사용 Speech-to-Text V2 RecognitionConfig.

    브라우저 MediaRecorder가 생성한 WebM/Opus 등은
    AutoDetectDecodingConfig로 감지한다.

    응급의료 용어는 inline PhraseSet으로 bias만 주며,
    없는 정보를 생성하지 않는다.
    """

    phrase_set = cloud_speech.PhraseSet(
        phrases=_google_stt_adaptation_phrase_entries()
    )

    adaptation = cloud_speech.SpeechAdaptation(
        phrase_sets=[
            cloud_speech.SpeechAdaptation.AdaptationPhraseSet(
                inline_phrase_set=phrase_set
            )
        ]
    )

    return cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["ko-KR"],
        model=GOOGLE_STT_MODEL,
        adaptation=adaptation,
        features=cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=True
        ),
    )


def transcribe_with_google_cloud(audio_content: bytes):
    project_id = get_google_cloud_project_id()

    speech_client = speech_v2.SpeechClient()

    request_obj = cloud_speech.RecognizeRequest(
        recognizer=(
            f"projects/{project_id}/"
            f"locations/{GOOGLE_STT_LOCATION}/"
            f"recognizers/_"
        ),
        config=build_google_stt_config(),
        content=audio_content,
    )

    response = speech_client.recognize(
        request=request_obj
    )

    transcript_parts = []
    confidences = []

    for result in response.results:
        if not result.alternatives:
            continue

        alternative = result.alternatives[0]

        text = str(
            alternative.transcript or ""
        ).strip()

        if text:
            transcript_parts.append(text)

        confidence = getattr(
            alternative,
            "confidence",
            None
        )

        if (
            confidence is not None
            and float(confidence) > 0
        ):
            confidences.append(
                float(confidence)
            )

    transcript = " ".join(
        transcript_parts
    ).strip()

    avg_confidence = (
        round(
            sum(confidences)
            / len(confidences),
            4
        )
        if confidences
        else None
    )

    return (
        transcript,
        avg_confidence,
        project_id,
    )


@app.route("/api/transcribe", methods=["POST"])
def transcribe_audio_api():
    """
    브라우저 MediaRecorder가 보낸 짧은 음성 파일을
    Google Cloud STT로 전사한다.

    음성 전사 기능만 담당하며
    LLM 분석/병원 추천/지도 로직은 호출하지 않는다.
    """

    try:
        if "audio" not in request.files:
            return jsonify({
                "success": False,
                "error": "audio 파일이 필요합니다."
            }), 400

        audio_file = request.files["audio"]
        audio_content = audio_file.read()

        if not audio_content:
            return jsonify({
                "success": False,
                "error": "녹음된 음성이 비어 있습니다."
            }), 400

        if (
            len(audio_content)
            > GOOGLE_STT_MAX_AUDIO_BYTES
        ):
            return jsonify({
                "success": False,
                "error": (
                    "녹음 파일이 너무 큽니다. "
                    "30초 이내로 다시 녹음해 주세요."
                ),
            }), 413

        (
            transcript,
            confidence,
            project_id,
        ) = transcribe_with_google_cloud(
            audio_content
        )

        if not transcript:
            return jsonify({
                "success": False,
                "error": (
                    "음성을 텍스트로 인식하지 못했습니다. "
                    "다시 녹음해 주세요."
                ),
            }), 422

        return jsonify({
            "success": True,
            "transcript": transcript,
            "confidence": confidence,
            "provider": "Google Cloud Speech-to-Text V2",
            "language_code": "ko-KR",
            "model": GOOGLE_STT_MODEL,
            "location": GOOGLE_STT_LOCATION,
            "project_id": project_id,
            "adaptation_phrase_count": _google_stt_adaptation_phrase_count(),
        })

    except Exception as e:
        print("GOOGLE STT ERROR:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# 3-2. Google Cloud Speech-to-Text 실시간 스트리밍 전사
# =========================================================

def build_google_streaming_stt_config(sample_rate_hz: int):
    """브라우저가 보내는 PCM16 mono 오디오용 실시간 STT 설정."""

    sample_rate_hz = int(sample_rate_hz)

    if sample_rate_hz < 8000 or sample_rate_hz > 96000:
        raise ValueError(f"지원하지 않는 샘플레이트입니다: {sample_rate_hz}Hz")

    phrase_set = cloud_speech.PhraseSet(
        phrases=_google_stt_adaptation_phrase_entries()
    )

    adaptation = cloud_speech.SpeechAdaptation(
        phrase_sets=[
            cloud_speech.SpeechAdaptation.AdaptationPhraseSet(
                inline_phrase_set=phrase_set
            )
        ]
    )

    recognition_config = cloud_speech.RecognitionConfig(
        explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
            encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate_hz,
            audio_channel_count=1,
        ),
        language_codes=["ko-KR"],
        model=GOOGLE_STREAMING_STT_MODEL,
        adaptation=adaptation,
        features=cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=True
        ),
    )

    return cloud_speech.StreamingRecognitionConfig(
        config=recognition_config,
        streaming_features=cloud_speech.StreamingRecognitionFeatures(
            interim_results=True
        ),
    )


def _safe_ws_send(ws, payload):
    try:
        ws.send(json.dumps(payload, ensure_ascii=False))
        return True
    except Exception:
        return False


def _is_expected_ws_close(error):
    """브라우저가 정상적으로 연결을 닫았을 때 생기는 1000/1001/1005는 오류로 취급하지 않는다."""
    message = str(error).lower()
    return (
        "connection closed" in message
        and any(code in message for code in ["1000", "1001", "1005"])
    )


@sock.route("/ws/transcribe")
def transcribe_streaming_ws(ws):
    """
    브라우저 마이크 PCM16 chunk를 WebSocket으로 받아
    Google Cloud STT V2 StreamingRecognize에 전달한다.
    """

    audio_queue = Queue()
    worker = None

    try:
        try:
            first_message = ws.receive()
        except Exception as e:
            if _is_expected_ws_close(e):
                return
            raise

        if first_message is None or isinstance(first_message, bytes):
            _safe_ws_send(
                ws,
                {"type": "error", "message": "STT 시작 설정이 없습니다."},
            )
            return

        start_data = json.loads(first_message)

        if start_data.get("type") != "start":
            _safe_ws_send(
                ws,
                {"type": "error", "message": "잘못된 STT 시작 요청입니다."},
            )
            return

        sample_rate = int(start_data.get("sampleRate") or 48000)
        project_id = get_google_cloud_project_id()
        recognizer = (
            f"projects/{project_id}/"
            f"locations/{GOOGLE_STREAMING_STT_LOCATION}/"
            "recognizers/_"
        )
        streaming_config = build_google_streaming_stt_config(sample_rate)

        def request_generator():
            yield cloud_speech.StreamingRecognizeRequest(
                recognizer=recognizer,
                streaming_config=streaming_config,
            )

            while True:
                chunk = audio_queue.get()

                if chunk is None:
                    break

                # Google streaming message 제한보다 작게 잘라 전송한다.
                for start_offset in range(0, len(chunk), 24000):
                    piece = chunk[start_offset:start_offset + 24000]
                    if piece:
                        yield cloud_speech.StreamingRecognizeRequest(audio=piece)

        def google_worker():
            try:
                speech_client = speech_v2.SpeechClient(
                    client_options=ClientOptions(
                        api_endpoint=f"{GOOGLE_STREAMING_STT_LOCATION}-speech.googleapis.com",
                        quota_project_id=project_id,
                    )
                )
                responses = speech_client.streaming_recognize(
                    requests=request_generator()
                )

                for response in responses:
                    for result in response.results:
                        if not result.alternatives:
                            continue

                        alternative = result.alternatives[0]
                        transcript = str(alternative.transcript or "").strip()

                        if not transcript:
                            continue

                        confidence = getattr(alternative, "confidence", None)
                        stability = getattr(result, "stability", None)

                        success = _safe_ws_send(
                            ws,
                            {
                                "type": "transcript",
                                "text": transcript,
                                "isFinal": bool(result.is_final),
                                "stability": (
                                    float(stability)
                                    if stability is not None
                                    else None
                                ),
                                "confidence": (
                                    float(confidence)
                                    if confidence is not None
                                    else None
                                ),
                            },
                        )

                        if not success:
                            return

                _safe_ws_send(ws, {"type": "done"})

            except Exception as e:
                print("GOOGLE STREAMING STT ERROR:", e)
                _safe_ws_send(
                    ws,
                    {"type": "error", "message": str(e)},
                )

        worker = Thread(target=google_worker, daemon=True)
        worker.start()

        _safe_ws_send(
            ws,
            {
                "type": "ready",
                "sampleRate": sample_rate,
                "model": GOOGLE_STREAMING_STT_MODEL,
                "location": GOOGLE_STREAMING_STT_LOCATION,
            },
        )

        while True:
            try:
                message = ws.receive()
            except Exception as e:
                if _is_expected_ws_close(e):
                    return
                raise

            if message is None:
                break

            if isinstance(message, bytes):
                if message:
                    audio_queue.put(message)
                continue

            try:
                command = json.loads(message)
            except Exception:
                continue

            if command.get("type") == "stop":
                audio_queue.put(None)

                if worker:
                    worker.join(timeout=15)

                return

    except Exception as e:
        if _is_expected_ws_close(e):
            return

        print("STREAMING WS ERROR:", e)
        _safe_ws_send(
            ws,
            {"type": "error", "message": str(e)},
        )

    finally:
        try:
            audio_queue.put_nowait(None)
        except Exception:
            pass


# =========================================================
# 4. Groq GPT-OSS 기반 환자 상태 분석
# =========================================================

def _groq_status_code(error):
    """Groq SDK 예외에서 HTTP status code를 가능한 범위에서 추출한다."""
    for attr in ("status_code", "code", "status"):
        value = getattr(error, attr, None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    return None


def _patient_cache_key(text):
    return " ".join(str(text or "").strip().split())

def get_cached_patient_analysis(text):
    key = _patient_cache_key(text)
    if not key:
        return None
    with PATIENT_ANALYSIS_CACHE_LOCK:
        cached = PATIENT_ANALYSIS_CACHE.get(key)
    if cached is None:
        return None
    # 내부 list를 호출자가 변경해도 캐시 원본이 바뀌지 않도록 복사한다.
    return json.loads(json.dumps(cached, ensure_ascii=False))

def cache_patient_analysis(text, patient):
    key = _patient_cache_key(text)
    if not key:
        return
    value = json.loads(json.dumps(patient, ensure_ascii=False))
    with PATIENT_ANALYSIS_CACHE_LOCK:
        PATIENT_ANALYSIS_CACHE[key] = value
        while len(PATIENT_ANALYSIS_CACHE) > PATIENT_ANALYSIS_CACHE_MAX:
            oldest_key = next(iter(PATIENT_ANALYSIS_CACHE))
            PATIENT_ANALYSIS_CACHE.pop(oldest_key, None)

def analyze_patient(text: str):
    cached = get_cached_patient_analysis(text)
    if cached is not None:
        print("GROQ ANALYSIS CACHE HIT: 동일 환자 문장 재사용")
        return cached

    system_prompt = """
너는 SAVER 응급환자 이송병원 탐색 시스템의 환자 상태 구조화 모듈이다.

목적은 확정 진단이나 특정 병원 추천이 아니라, 입력된 환자 서술을 SAVER 병원 데이터베이스와 비교 가능한 구조화된 요구조건으로 변환하는 것이다.

반드시 아래 규칙을 지켜라.

[기본 규칙]
1. 모든 문자열 출력은 반드시 한국어로 작성한다.
2. 입력에 없는 증상, 나이, 성별, 신체 좌우, 활력징후를 임의로 추가하지 않는다.
3. 특정 병원명을 출력하거나 추천하지 않는다.
4. 각 필드는 서로 논리적으로 일관되게 판단한다.

[KTAS]
5. ktas는 1~5 사이 정수만 사용한다.
6. KTAS의 방향은 반드시 다음과 같이 해석한다.
   - 1: 가장 긴급
   - 2: 매우 긴급
   - 3: 긴급
   - 4: 덜 긴급
   - 5: 비응급
7. 환자 서술의 중증도 및 다른 Boolean 요구조건과 모순되지 않도록 결정한다.

[전문의/진료과]
8. specialists에는 병원 DB와 매칭할 수 있도록 대한민국 의료기관에서 사용하는 표준 한국어 진료과명을 작성한다.
   예: 응급의학과, 소아청소년과, 내과, 신경과, 신경외과, 외과, 정형외과, 심장내과, 순환기내과, 심장혈관흉부외과, 산부인과.
9. 영문 진료과명은 사용하지 않는다. 예를 들어 Pediatrics가 아니라 소아청소년과, Emergency Medicine이 아니라 응급의학과로 작성한다.
10. 환자에게 실제로 필요한 진료과만 포함하며 불필요하게 많은 진료과를 추가하지 않는다.

[의료장비]
11. equipment에는 SAVER 병원 DB에서 직접 비교 가능한 장비만 작성한다: CT, MRI, CAG, 혈관조영, 인공호흡기.
12. 수액, 수액백, infusion pump, vital sign monitor, 일반적인 처치도구나 소모품은 equipment에 넣지 않는다.
13. 위 장비가 특별히 필요하지 않다면 빈 배열 []을 반환한다.

[병원 자원]
14. req_icu는 중환자실이 실제로 필요하다고 판단되는 경우에만 true이다.
15. req_or는 수술실이 실제로 필요한 경우에만 true이다.
16. req_er_bed는 응급실에서의 수용 및 처치가 필요한 경우 true이다.
17. req_severe_acceptance는 중증환자 수용 가능 여부의 확인이 필요한 경우 true이다.

[기타]
18. suspected_category는 간결한 한국어 임상 범주로 작성하되, 병원 선택에 영향을 줄 수 있는 핵심 수식어를 불필요하게 생략하지 않는다.
    - 입력에서 소아·신생아·영아 등 연령군이 명확하게 확인되면 해당 연령군을 보존한다.
    - 입력에서 중증·경증, 급성 등 중증도 또는 시간적 특성이 명확하게 표현되어 있고 병원 선택에 의미가 있으면 이를 보존한다.
    - 예: "5세 남아, 중증 탈수 의심, 소아 응급 진료 필요" → "소아 중증 탈수"
    - 예: 단순히 "탈수 의심"이라고만 제시된 경우 입력에 없는 "소아"나 "중증"을 임의로 추가하지 않는다.
    - 너무 포괄적인 단어 하나만 남겨 원문의 중요한 임상적 구분을 소실하지 않는다.
19. golden_time_min은 이송의 시간적 우선순위를 나타내는 분 단위 정수로 작성한다.
20. 반드시 지정된 JSON schema만 반환한다.
""".strip()

    response = GROQ_CLIENT.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(text or "").strip()},
        ],
        reasoning_effort="medium",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "patient_info",
                "strict": True,
                "schema": PatientInfo.model_json_schema(),
            },
        },
    )

    content = response.choices[0].message.content or ""
    if not content.strip():
        raise RuntimeError("Groq 환자 상태 구조화 응답이 비어 있습니다.")

    patient = PatientInfo.model_validate_json(content)
    result = patient.model_dump()
    cache_patient_analysis(text, result)

    print(f"GROQ ANALYSIS MODEL: {GROQ_MODEL}")
    return result


@app.route("/api/analyze-patient", methods=["POST"])
def analyze_patient_api():
    try:
        data = request.get_json()

        if (
            not data
            or "text" not in data
        ):
            return jsonify({
                "success": False,
                "error": "text가 필요합니다."
            }), 400

        result = analyze_patient(
            data["text"]
        )

        return jsonify({
            "success": True,
            "patient": result
        })

    except Exception as e:
        print("AI ERROR:", e)
        status_code = _groq_status_code(e)
        if status_code == 429:
            return jsonify({
                "success": False,
                "error": "Groq 무료 API 요청 한도에 도달했습니다. 한도가 복구된 뒤 다시 시도해 주세요."
            }), 429
        if status_code is not None and status_code >= 500:
            return jsonify({
                "success": False,
                "error": "Groq 모델 서버에서 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
            }), 503

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# 5. 공통 변환 함수
# =========================================================

def to_number(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    value = str(value).strip()

    if (
        value == ""
        or value.lower() in [
            "none",
            "nan"
        ]
    ):
        return None

    try:
        return float(value)

    except ValueError:
        return None


def numeric_available(value):
    num = to_number(value)

    if num is None:
        return None

    return num > 0


def yn_to_bool(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    value = str(value).strip().upper()

    if (
        value == ""
        or value in [
            "NONE",
            "NAN"
        ]
    ):
        return None

    if value.startswith("Y"):
        return True

    if value.startswith("N"):
        return False

    return None


def normalize_yes_no(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    value = str(value).strip().upper()

    if value == "":
        return None

    if value in [
        "Y",
        "YES",
        "TRUE",
        "1",
        "있음",
    ]:
        return True

    if value in [
        "N",
        "NO",
        "FALSE",
        "0",
        "없음",
    ]:
        return False

    return None


def get_hospital_name(hospital):
    for col in [
        "hospital_name",
        "target_hospital",
        "dutyname",
        "dutyName",
        "hospital_display_name",
        "name",
    ]:
        if (
            col in hospital
            and str(
                hospital.get(col)
            ).strip()
        ):
            return str(
                hospital.get(col)
            ).strip()

    return "병원명 미확인"


def text_contains_any(
    text,
    keywords
):
    text = str(text or "")

    return any(
        keyword in text
        for keyword in keywords
    )


# =========================================================
# 5-1. 거리 / ETA / 병원 혼잡도 계산 (알고리즘 모델 공통 입력값)
# =========================================================

# 구급차 평균 주행 속도(km/h) 가정치.
# 카카오내비 실시간 교통 연동 전까지 이동시간을
# "시뮬레이션"하기 위한 값이다.
AVG_AMBULANCE_SPEED_KMH = 40.0

# 출동/하차 등 순수 주행 이외에 걸리는 고정 오버헤드(분)
FIXED_DISPATCH_OVERHEAD_MIN = 3.0


def haversine_km(
    lat1,
    lng1,
    lat2,
    lng2
):
    """
    두 좌표 사이의 직선거리(km)를
    하버사인 공식으로 계산한다.
    """

    try:
        (
            lat1,
            lng1,
            lat2,
            lng2,
        ) = map(
            float,
            [
                lat1,
                lng1,
                lat2,
                lng2,
            ]
        )

    except (
        TypeError,
        ValueError
    ):
        return None

    R = 6371.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(
        lat2 - lat1
    )

    dlambda = math.radians(
        lng2 - lng1
    )

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return R * c


def is_valid_korean_coordinate(
    lat,
    lng
):
    try:
        lat = float(lat)
        lng = float(lng)

    except (
        TypeError,
        ValueError
    ):
        return False

    return (
        math.isfinite(lat)
        and math.isfinite(lng)
        and 33 <= lat <= 39.5
        and 124 <= lng <= 132
    )


def get_hospital_lat_lng(
    hospital
):
    lat = to_number(
        hospital.get("latitude")
    )

    lng = to_number(
        hospital.get("longitude")
    )

    if lat is None:
        lat = to_number(
            hospital.get("lat")
        )

    if lng is None:
        lng = to_number(
            hospital.get("lng")
        )

    if (
        lat is None
        or lng is None
    ):
        return None, None

    return lat, lng


def call_kakao_route(
    origin_lat,
    origin_lng,
    destination_lat,
    destination_lng
):
    if not KAKAO_REST_API_KEY:
        raise RuntimeError(
            ".env에 KAKAO_REST_API_KEY가 "
            "설정되어 있지 않습니다."
        )

    if not is_valid_korean_coordinate(
        origin_lat,
        origin_lng
    ):
        raise ValueError(
            "출발지 좌표가 올바르지 않습니다."
        )

    if not is_valid_korean_coordinate(
        destination_lat,
        destination_lng
    ):
        raise ValueError(
            "목적지 좌표가 올바르지 않습니다."
        )

    params = {
        "origin": (
            f"{float(origin_lng)},"
            f"{float(origin_lat)}"
        ),
        "destination": (
            f"{float(destination_lng)},"
            f"{float(destination_lat)}"
        ),
        "priority": "TIME",
        "alternatives": "false",
        "road_details": "true",
        "roadevent": "0",
    }

    response = requests.get(
        "https://apis-navi.kakaomobility.com/v1/directions",
        params=params,
        headers={
            "Authorization": (
                f"KakaoAK "
                f"{KAKAO_REST_API_KEY}"
            ),
            "Content-Type": (
                "application/json"
            ),
        },
        timeout=KAKAO_REQUEST_TIMEOUT_SEC,
    )

    try:
        data = response.json()

    except Exception:
        data = {
            "raw": response.text[:1000]
        }

    if response.status_code >= 400:
        raise RuntimeError(
            "카카오모빌리티 길찾기 API 오류: "
            f"HTTP {response.status_code} / "
            f"{data}"
        )

    return data


def random_location_near_hospital(
    hospital
):
    (
        hosp_lat,
        hosp_lng,
    ) = get_hospital_lat_lng(
        hospital
    )

    if (
        hosp_lat is None
        or hosp_lng is None
    ):
        return None

    radius_km = random.uniform(
        PATIENT_MIN_RADIUS_KM,
        PATIENT_MAX_RADIUS_KM
    )

    bearing = random.uniform(
        0,
        2 * math.pi
    )

    delta_lat = (
        radius_km / 111.0
    ) * math.cos(
        bearing
    )

    delta_lng = (
        radius_km
        / (
            111.0
            * math.cos(
                math.radians(
                    hosp_lat
                )
            )
        )
    ) * math.sin(
        bearing
    )

    return {
        "lat": round(
            hosp_lat + delta_lat,
            6
        ),
        "lng": round(
            hosp_lng + delta_lng,
            6
        ),
        "anchor_hospital_name": (
            get_hospital_name(
                hospital
            )
        ),
        "anchor_hospital_lat": (
            hosp_lat
        ),
        "anchor_hospital_lng": (
            hosp_lng
        ),
        "radius_km": round(
            radius_km,
            2
        ),
    }


def compute_distance_and_eta(
    hospital,
    ambulance_lat,
    ambulance_lng
):
    """
    병원 위경도와 구급차 현재 위치 사이의 직선거리(km)와,
    평균 주행속도 가정치를 이용한 시뮬레이션 ETA(분)를 계산한다.

    실제 도로 경로/실시간 교통정보 기반 ETA가 아니라
    직선거리 기반 근사치이다.
    """

    (
        hosp_lat,
        hosp_lng,
    ) = get_hospital_lat_lng(
        hospital
    )

    if (
        hosp_lat is None
        or hosp_lng is None
    ):
        return None, None

    distance_km = haversine_km(
        ambulance_lat,
        ambulance_lng,
        hosp_lat,
        hosp_lng
    )

    if distance_km is None:
        return None, None

    # 직선거리는 실제 도로거리보다 짧으므로
    # 보정계수 1.3을 곱해 근사한다.
    road_distance_km = (
        distance_km * 1.3
    )

    eta_min = (
        (
            road_distance_km
            / AVG_AMBULANCE_SPEED_KMH
        )
        * 60
        + FIXED_DISPATCH_OVERHEAD_MIN
    )

    return (
        round(
            road_distance_km,
            2
        ),
        round(
            eta_min,
            1
        )
    )


def compute_congestion_score(
    hospital
):
    """
    병원 혼잡도(0~1, 1에 가까울수록 혼잡)를
    실시간 가용병상 수 기반으로 근사한다.
    """

    hvec = to_number(
        hospital.get("hvec")
    )

    if hvec is None:
        return None

    hvec = max(
        0.0,
        hvec
    )

    reference_capacity = 5.0

    congestion = (
        1.0
        - min(
            hvec,
            reference_capacity
        )
        / reference_capacity
    )

    return round(
        congestion,
        3
    )


def compute_data_freshness_score(hospital, now=None):
    """실시간 데이터의 최신성을 0~1로 환산한다."""
    timestamp = None
    for col in ["data_fetched_at", "hvidate", "master_last_updated_at", "departments_last_checked_at"]:
        value = hospital.get(col)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue

        text = str(value).strip()
        if not text or text.lower() in ["none", "nan"]:
            continue

        # 응급의료 API hvidate의 YYYYMMDDHHMMSS 숫자 형식 보정.
        compact = text.split(".")[0]
        if compact.isdigit() and len(compact) == 14:
            parsed = pd.to_datetime(compact, format="%Y%m%d%H%M%S", errors="coerce")
        else:
            parsed = pd.to_datetime(text, errors="coerce")

        if not pd.isna(parsed):
            timestamp = parsed
            break

    if timestamp is None:
        return 0.2

    now_ts = pd.Timestamp(now) if now is not None else pd.Timestamp.now()
    if getattr(timestamp, "tzinfo", None) is not None and now_ts.tzinfo is None:
        timestamp = timestamp.tz_localize(None)
    elif getattr(timestamp, "tzinfo", None) is None and now_ts.tzinfo is not None:
        now_ts = now_ts.tz_localize(None)

    age_min = max(0.0, (now_ts - timestamp).total_seconds() / 60.0)
    if age_min <= 15:
        return 1.0
    if age_min <= 60:
        return 0.85
    if age_min <= 180:
        return 0.65
    if age_min <= 720:
        return 0.4
    if age_min <= 1440:
        return 0.25
    return 0.1

def get_patient_priority_profile(patient):
    """환자 중증도와 골든타임에 따라 모델의 선호도 프로파일을 반환한다.

    KTAS 1~2 또는 골든타임 60분 이하는 시간 민감 환자로 보고,
    예상 치료시작 지연과 임상 적합성에 더 큰 비중을 둔다.
    """
    ktas = to_number(patient.get("ktas"))
    golden_time = to_number(patient.get("golden_time_min"))
    critical = (ktas is not None and ktas <= 2) or (golden_time is not None and golden_time <= 60)
    profile_name = "critical" if critical else "standard"

    if ktas is not None:
        severity_weight = {1: 3.0, 2: 2.4, 3: 1.7, 4: 1.2, 5: 1.0}.get(int(ktas), 1.5)
    else:
        severity_weight = 1.5

    return {
        "name": profile_name,
        "label": "시간 민감/중증" if critical else "표준",
        "severity_weight": severity_weight,
        "golden_time_min": golden_time,
        "topsis_weights": TOPSIS_VARIANTS["C2"]["weight_profiles"][profile_name],
    }

def compute_operational_estimates(hospital, eta_min, congestion_score, now=None):
    """
    현재 보유한 데이터로 예상 대기·거절·재이송 지연을 보수적으로 근사한다.

    해당 값은 학습된 확률이 아니므로, 실제 병원 수용 응답/도착 기록이
    쌓이면 교체해야 하는 프록시이다.
    """
    acceptance_score = float(to_number(hospital.get("acceptance_score")) or 0.0)
    clinical_score = min(1.0, max(0.0, acceptance_score / 100.0))
    freshness = compute_data_freshness_score(hospital, now=now)
    realtime_ok = str(hospital.get("realtime_match_status", "")).strip() == "OK"

    unknown_reasons = hospital.get("unknown_reasons") or []
    if isinstance(unknown_reasons, str):
        unknown_count = len([item for item in unknown_reasons.split("/") if item.strip()])
    else:
        unknown_count = len(unknown_reasons)

    reliability = 0.7 * freshness + 0.3 * (1.0 if realtime_ok else 0.3)
    reliability = min(1.0, max(0.0, reliability - min(0.3, unknown_count * 0.06)))

    estimated_acceptance_probability = (
        0.35 + 0.45 * clinical_score + 0.20 * reliability - min(0.25, unknown_count * 0.05)
    )
    estimated_acceptance_probability = min(0.98, max(0.05, estimated_acceptance_probability))

    congestion = 0.5 if congestion_score is None else min(1.0, max(0.0, float(congestion_score)))
    expected_wait = MIN_EXPECTED_WAIT_MIN + CONGESTION_WAIT_RANGE_MIN * congestion
    rejection_delay = (1.0 - estimated_acceptance_probability) * ASSUMED_REJECTION_RESEARCH_MIN
    retransfer_delay = (
        max(0.0, 0.85 - clinical_score) / 0.85 * ASSUMED_RETRANSFER_MIN
        + (1.0 - reliability) * 5.0
    )

    effective_delay = None
    if eta_min is not None:
        effective_delay = float(eta_min) + expected_wait + rejection_delay + retransfer_delay

    return {
        "data_freshness_score": round(freshness, 3),
        "data_reliability_score": round(reliability, 3),
        "estimated_acceptance_probability": round(estimated_acceptance_probability, 3),
        "expected_wait_min": round(expected_wait, 1),
        "expected_rejection_delay_min": round(rejection_delay, 1),
        "expected_retransfer_delay_min": round(retransfer_delay, 1),
        "effective_treatment_delay_min": round(effective_delay, 1) if effective_delay is not None else None,
    }

def compute_specialty_margin(hospital, patient):
    """
    필수 전문과를 충족한 후 남는 전문의 수를 '전문치료 여유도'로 계산한다.

    병상(hvec)은 congestion_score와 중복되므로 포함하지 않는다.
    전문의 인원 자료가 없고 진료과 존재 여부만 있는 병원은 추가 여유를
    증명할 수 없으므로 0으로 둔다. 결측을 임의의 장점으로 바꾸지 않기 위한 보수적 정의다.
    """
    required_specialties = {
        str(name).strip()
        for name in (patient or {}).get("specialists", []) or []
        if str(name).strip()
    }
    if not required_specialties:
        return 0.0, "필수 전문과 요구 없음"

    total_count = to_number(hospital.get("required_specialist_count_total"))
    if total_count is None or total_count <= 0:
        return 0.0, "필수 전문과 인원수 미확인"

    minimum_required_count = float(len(required_specialties))
    margin = max(0.0, float(total_count) - minimum_required_count)
    return round(margin, 3), f"전문의 {int(total_count)}명 - 필수과 최소 {int(minimum_required_count)}명"

def get_topsis_criteria(variant=DEFAULT_TOPSIS_VARIANT, patient=None):
    """변형별 기준을 복사해 반환한다. 원본 설정은 절대 변경하지 않는다."""
    variant = str(variant or DEFAULT_TOPSIS_VARIANT).strip().upper()
    if variant not in TOPSIS_VARIANTS:
        raise ValueError(f"알 수 없는 TOPSIS 변형입니다: {variant}")

    config = TOPSIS_VARIANTS[variant]
    if config.get("criteria_from"):
        base = TOPSIS_VARIANTS[config["criteria_from"]]["criteria"]
    else:
        base = config["criteria"]

    criteria = {
        name: {"benefit": bool(values["benefit"]), "weight": float(values["weight"])}
        for name, values in base.items()
    }

    profile = None
    if config.get("patient_aware"):
        profile = get_patient_priority_profile(patient or {})
        profile_weights = config["weight_profiles"][profile["name"]]
        for name in criteria:
            criteria[name]["weight"] = float(profile_weights[name])

    weight_sum = sum(values["weight"] for values in criteria.values())
    if not math.isclose(weight_sum, 1.0, abs_tol=1e-9):
        raise ValueError(f"{variant} TOPSIS 가중치 합이 1이 아닙니다: {weight_sum}")

    return criteria, profile

def get_topsis_variant_metadata(variant=DEFAULT_TOPSIS_VARIANT, patient=None):
    """API와 실험 로그에 내보낼 재현 가능한 변형 메타데이터."""
    variant = str(variant or DEFAULT_TOPSIS_VARIANT).strip().upper()
    criteria, profile = get_topsis_criteria(variant, patient)
    config = TOPSIS_VARIANTS[variant]
    return {
        "key": variant,
        "label": config["label"],
        "description": config["description"],
        "experimental": bool(config.get("experimental", False)),
        "patient_profile": profile["name"] if profile else None,
        "criteria": [
            {
                "name": name,
                "direction": "benefit" if values["benefit"] else "cost",
                "weight": values["weight"],
            }
            for name, values in criteria.items()
        ],
    }

def _topsis_closeness(df, criteria):
    """
    변형을 모르는 공통 TOPSIS 순수 계산 엔진.

    결측치는 benefit 기준의 최솟값, cost 기준의 최댓값으로
    보수적 대체한다. 변형 분기는 이 함수 안에 두지 않는다.
    """
    if df.empty:
        return pd.Series(dtype=float, index=df.index)

    missing_columns = [name for name in criteria if name not in df.columns]
    if missing_columns:
        raise KeyError(f"TOPSIS 기준 컬럼이 없습니다: {', '.join(missing_columns)}")

    matrix = pd.DataFrame(index=df.index)
    for column, config in criteria.items():
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            fill_value = values.min() if config["benefit"] else values.max()
        else:
            fill_value = 0.0
        matrix[column] = values.fillna(fill_value)

    vector_norm = np.sqrt((matrix ** 2).sum()).replace(0, np.nan)
    normalized = matrix.divide(vector_norm, axis=1).fillna(0.0)
    weighted = normalized.copy()
    for column, config in criteria.items():
        weighted[column] = normalized[column] * float(config["weight"])

    ideal_best = {}
    ideal_worst = {}
    for column, config in criteria.items():
        if config["benefit"]:
            ideal_best[column] = weighted[column].max()
            ideal_worst[column] = weighted[column].min()
        else:
            ideal_best[column] = weighted[column].min()
            ideal_worst[column] = weighted[column].max()

    distance_best = np.sqrt(sum((weighted[column] - ideal_best[column]) ** 2 for column in criteria))
    distance_worst = np.sqrt(sum((weighted[column] - ideal_worst[column]) ** 2 for column in criteria))
    denominator = (distance_best + distance_worst).replace(0, np.nan)
    return (distance_worst / denominator).fillna(0.5)

def attach_routing_fields(
    hospital_df,
    ambulance_lat,
    ambulance_lng
):
    """
    acceptance/rejection 계산이 끝난 결과 DataFrame에
    거리/ETA/혼잡도 컬럼을 추가한다.
    """

    if hospital_df.empty:
        hospital_df["distance_km"] = []
        hospital_df["eta_min"] = []
        hospital_df["congestion_score"] = []

        return hospital_df

    distances = []
    etas = []
    congestions = []

    for _, row in hospital_df.iterrows():
        (
            distance_km,
            eta_min,
        ) = compute_distance_and_eta(
            row,
            ambulance_lat,
            ambulance_lng
        )

        distances.append(
            distance_km
        )

        etas.append(
            eta_min
        )

        congestions.append(
            compute_congestion_score(
                row
            )
        )

    hospital_df = (
        hospital_df.copy()
    )

    hospital_df[
        "distance_km"
    ] = distances

    hospital_df[
        "eta_min"
    ] = etas

    hospital_df[
        "congestion_score"
    ] = congestions

    return hospital_df


# =========================================================
# 5-2. 알고리즘 모델 A/B/C/D (최적 병원 순위 산정)
# =========================================================

def _normalize_series(
    series,
    higher_is_better
):
    """
    0~1 min-max 정규화.
    값이 전부 같거나 없으면 중립값 0.5로 채운다.
    """

    values = series.astype(float)

    valid = values.dropna()

    if (
        valid.empty
        or valid.max()
        == valid.min()
    ):
        return pd.Series(
            [0.5] * len(values),
            index=values.index
        )

    normalized = (
        values
        - valid.min()
    ) / (
        valid.max()
        - valid.min()
    )

    normalized = normalized.fillna(
        0.5
    )

    if not higher_is_better:
        normalized = 1 - normalized

    return normalized



def _preselect_kakao_candidates(hospital_df, ambulance_lat, ambulance_lng, limit=KAKAO_RANKING_CANDIDATE_LIMIT):
    """카카오 API 호출 대상만 최대 30개로 줄인다. 이 거리값은 최종 모델/화면에 사용하지 않는다."""
    if hospital_df.empty:
        return hospital_df.copy()

    df = hospital_df.copy()
    preselect_distances = []
    for _, row in df.iterrows():
        hosp_lat, hosp_lng = get_hospital_lat_lng(row)
        if hosp_lat is None or hosp_lng is None:
            preselect_distances.append(float("inf"))
            continue
        distance = haversine_km(ambulance_lat, ambulance_lng, hosp_lat, hosp_lng)
        preselect_distances.append(float(distance) if distance is not None else float("inf"))

    df["_kakao_preselect_distance"] = preselect_distances
    df = df[np.isfinite(pd.to_numeric(df["_kakao_preselect_distance"], errors="coerce"))].copy()
    df = df.sort_values("_kakao_preselect_distance", ascending=True, kind="mergesort").head(int(limit)).copy()
    return df.drop(columns=["_kakao_preselect_distance"], errors="ignore")


def _kakao_route_cache_key(origin_lat, origin_lng, destination_lat, destination_lng):
    return (
        round(float(origin_lat), 6),
        round(float(origin_lng), 6),
        round(float(destination_lat), 6),
        round(float(destination_lng), 6),
    )


def _get_cached_kakao_eta(origin_lat, origin_lng, destination_lat, destination_lng):
    """같은 출발지/도착지의 성공·실패 라우팅 상태를 TTL 동안 재사용한다."""
    key = _kakao_route_cache_key(origin_lat, origin_lng, destination_lat, destination_lng)
    with KAKAO_ROUTE_CACHE_LOCK:
        item = KAKAO_ROUTE_CACHE.get(key)
        if not item:
            return None
        if time.time() - float(item.get("cached_at", 0)) > KAKAO_ROUTE_CACHE_TTL_SEC:
            KAKAO_ROUTE_CACHE.pop(key, None)
            return None
        return dict(item)


def _set_cached_kakao_eta(origin_lat, origin_lng, destination_lat, destination_lng, distance_km, eta_min):
    key = _kakao_route_cache_key(origin_lat, origin_lng, destination_lat, destination_lng)
    with KAKAO_ROUTE_CACHE_LOCK:
        KAKAO_ROUTE_CACHE[key] = {
            "kakao_route_ok": True,
            "distance_km": float(distance_km),
            "eta_min": int(eta_min),
            "cached_at": time.time(),
        }


def _set_cached_kakao_failure(origin_lat, origin_lng, destination_lat, destination_lng):
    """다중 목적지에서 확인되지 않은 후보도 잠시 캐시해 모델 전환 시 재요청을 막는다."""
    key = _kakao_route_cache_key(origin_lat, origin_lng, destination_lat, destination_lng)
    with KAKAO_ROUTE_CACHE_LOCK:
        KAKAO_ROUTE_CACHE[key] = {
            "kakao_route_ok": False,
            "cached_at": time.time(),
        }


def call_kakao_multi_destination_eta(hospital_df, ambulance_lat, ambulance_lng):
    """
    카카오 다중 목적지 API만 사용해 최대 30개 후보의 도로거리/ETA를 조회한다.

    - 성공한 후보는 실제 카카오 거리/ETA를 저장한다.
    - result_code 실패/반경 초과 후보는 여기서 single directions를 추가 호출하지 않는다.
    - 성공/실패 모두 TTL 캐시에 저장하므로 같은 위치에서 A/B/C/D를 바꿔도
      동일 후보에 외부 API를 반복 호출하지 않는다.
    """
    if hospital_df.empty:
        return hospital_df.copy()
    if len(hospital_df) > KAKAO_RANKING_CANDIDATE_LIMIT:
        raise ValueError(f"카카오 ETA 후보는 최대 {KAKAO_RANKING_CANDIDATE_LIMIT}개까지 가능합니다.")
    if not KAKAO_REST_API_KEY:
        raise RuntimeError(".env에 KAKAO_REST_API_KEY가 설정되어 있지 않습니다.")
    if not is_valid_korean_coordinate(ambulance_lat, ambulance_lng):
        raise ValueError("환자 위치 좌표가 올바르지 않습니다.")

    out = hospital_df.copy()
    out["distance_km"] = np.nan
    out["eta_min"] = np.nan
    out["kakao_route_ok"] = False
    out["routing_source"] = "kakao_unavailable"

    uncached = []
    for idx, row in hospital_df.iterrows():
        lat, lng = get_hospital_lat_lng(row)
        if not is_valid_korean_coordinate(lat, lng):
            continue

        cached = _get_cached_kakao_eta(ambulance_lat, ambulance_lng, lat, lng)
        if cached is not None:
            if bool(cached.get("kakao_route_ok")):
                out.at[idx, "distance_km"] = round(float(cached["distance_km"]), 2)
                out.at[idx, "eta_min"] = int(cached["eta_min"])
                out.at[idx, "kakao_route_ok"] = True
                out.at[idx, "routing_source"] = "kakao_cache"
            else:
                out.at[idx, "routing_source"] = "kakao_unavailable_cached"
            continue

        uncached.append((idx, float(lat), float(lng)))

    if not uncached:
        return out

    destinations = []
    key_to_item = {}
    for pos, (idx, lat, lng) in enumerate(uncached):
        key = str(pos)
        destinations.append({"key": key, "x": lng, "y": lat})
        key_to_item[key] = (idx, lat, lng)

    response = requests.post(
        "https://apis-navi.kakaomobility.com/v1/destinations/directions",
        headers={
            "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "origin": {"x": float(ambulance_lng), "y": float(ambulance_lat)},
            "destinations": destinations,
            "radius": 10000,
            "priority": "TIME",
            "roadevent": 0,
        },
        timeout=KAKAO_REQUEST_TIMEOUT_SEC,
    )

    try:
        data = response.json()
    except Exception:
        data = {"raw": response.text[:1000]}

    if response.status_code >= 400:
        raise RuntimeError(
            f"카카오모빌리티 다중 목적지 ETA 조회 실패 (HTTP {response.status_code}): {data}"
        )

    route_by_key = {str(route.get("key")): route for route in (data.get("routes") or [])}
    for key, (idx, lat, lng) in key_to_item.items():
        route = route_by_key.get(key) or {}
        if int(route.get("result_code", -1)) != 0:
            _set_cached_kakao_failure(ambulance_lat, ambulance_lng, lat, lng)
            continue

        summary = route.get("summary") or {}
        distance_m = to_number(summary.get("distance"))
        duration_sec = to_number(summary.get("duration"))
        if distance_m is None or duration_sec is None:
            _set_cached_kakao_failure(ambulance_lat, ambulance_lng, lat, lng)
            continue

        distance_km = round(float(distance_m) / 1000.0, 2)
        eta_min = max(1, int(round(float(duration_sec) / 60.0)))
        out.at[idx, "distance_km"] = distance_km
        out.at[idx, "eta_min"] = eta_min
        out.at[idx, "kakao_route_ok"] = True
        out.at[idx, "routing_source"] = "kakao_multi_destination"
        _set_cached_kakao_eta(
            ambulance_lat,
            ambulance_lng,
            lat,
            lng,
            distance_km,
            eta_min,
        )

    return out


def _attach_estimated_route_fallback(hospital_df, ambulance_lat, ambulance_lng):
    """
    Kakao 다중 목적지 결과가 없는 병원도 후보에서 제거하지 않는다.

    해당 병원은 기존 SAVER의 직선거리→도로거리 보정/평균속도 ETA 추정값을 사용하며,
    routing_source='estimated_fallback'으로 명확히 표시한다.
    """
    if hospital_df.empty:
        return hospital_df.copy()

    out = hospital_df.copy()
    if "distance_km" not in out.columns:
        out["distance_km"] = np.nan
    if "eta_min" not in out.columns:
        out["eta_min"] = np.nan
    if "kakao_route_ok" not in out.columns:
        out["kakao_route_ok"] = False
    if "routing_source" not in out.columns:
        out["routing_source"] = "kakao_unavailable"

    for idx, row in out.iterrows():
        if bool(row.get("kakao_route_ok")):
            continue

        distance_km, eta_min = compute_distance_and_eta(row, ambulance_lat, ambulance_lng)
        if distance_km is None or eta_min is None:
            continue

        out.at[idx, "distance_km"] = distance_km
        out.at[idx, "eta_min"] = eta_min
        out.at[idx, "routing_source"] = "estimated_fallback"

    return out


def _prepare_routing_candidates_two_batches(
    acceptable_df,
    ambulance_lat,
    ambulance_lng,
    target_confirmed=KAKAO_TARGET_CONFIRMED_ROUTES,
):
    """
    Hard Filter 통과 병원 중 가까운 후보를 최대 2개 batch로 조회한다.

    1차 batch(최대 30개)에서 실제 Kakao 경로가 target_confirmed개 이상 확보되면 종료한다.
    부족할 때만 다음 30개를 2차 batch로 조회한다.
    각 batch의 Kakao 누락 후보는 제거하지 않고 추정 거리/ETA를 붙인다.
    """
    if acceptable_df.empty:
        return acceptable_df.copy(), 0, 0

    # 기존 _preselect_kakao_candidates와 동일한 직선거리 사전선별 규칙을 사용하되,
    # 최대 2 batch 분량까지 한 번만 정렬한다.
    df = acceptable_df.copy()
    preselect_distances = []
    for _, row in df.iterrows():
        hosp_lat, hosp_lng = get_hospital_lat_lng(row)
        if not is_valid_korean_coordinate(hosp_lat, hosp_lng):
            preselect_distances.append(float("inf"))
            continue
        distance = haversine_km(ambulance_lat, ambulance_lng, hosp_lat, hosp_lng)
        preselect_distances.append(float(distance) if distance is not None else float("inf"))

    df["_kakao_preselect_distance"] = preselect_distances
    finite_mask = np.isfinite(pd.to_numeric(df["_kakao_preselect_distance"], errors="coerce"))
    df = df[finite_mask].copy()
    df = df.sort_values("_kakao_preselect_distance", ascending=True, kind="mergesort")

    max_candidates = KAKAO_RANKING_CANDIDATE_LIMIT * KAKAO_MAX_BATCHES
    df = df.head(max_candidates).drop(columns=["_kakao_preselect_distance"], errors="ignore")

    routed_batches = []
    confirmed_count = 0
    batch_count = 0

    for batch_no in range(KAKAO_MAX_BATCHES):
        start = batch_no * KAKAO_RANKING_CANDIDATE_LIMIT
        end = start + KAKAO_RANKING_CANDIDATE_LIMIT
        batch = df.iloc[start:end].copy()
        if batch.empty:
            break

        # 2차 batch는 1차에서 실제 Kakao 경로가 충분하지 않을 때만 호출한다.
        if batch_no > 0 and confirmed_count >= min(int(target_confirmed), len(df)):
            break

        batch_count += 1
        try:
            routed = call_kakao_multi_destination_eta(batch, ambulance_lat, ambulance_lng)
        except Exception as exc:
            print(f"KAKAO MULTI BATCH ERROR batch={batch_no + 1}: {exc}")
            routed = batch.copy()
            routed["distance_km"] = np.nan
            routed["eta_min"] = np.nan
            routed["kakao_route_ok"] = False
            routed["routing_source"] = "kakao_batch_error"

        confirmed_count += int(pd.Series(routed["kakao_route_ok"]).fillna(False).astype(bool).sum())
        routed = _attach_estimated_route_fallback(routed, ambulance_lat, ambulance_lng)
        routed_batches.append(routed)

    if not routed_batches:
        return acceptable_df.head(0).copy(), 0, 0

    combined = pd.concat(routed_batches, axis=0)
    combined = combined[~combined.index.duplicated(keep="first")].copy()
    return combined, confirmed_count, batch_count


def attach_model_feature_fields(hospital_df, patient=None, as_of=None):
    """카카오 거리/ETA가 붙은 후보에 개선 ver_0920의 모델용 파생변수를 추가한다."""
    if hospital_df.empty:
        return hospital_df.copy()

    df = hospital_df.copy()
    congestions = []
    operational_estimates = []
    specialty_margins = []
    specialty_margin_bases = []

    for _, row in df.iterrows():
        congestion_score = compute_congestion_score(row)
        congestions.append(congestion_score)
        operational_estimates.append(
            compute_operational_estimates(row, to_number(row.get("eta_min")), congestion_score, now=as_of)
        )
        specialty_margin, specialty_margin_basis = compute_specialty_margin(row, patient or {})
        specialty_margins.append(specialty_margin)
        specialty_margin_bases.append(specialty_margin_basis)

    df["congestion_score"] = congestions
    if operational_estimates:
        for col in operational_estimates[0]:
            df[col] = [estimate[col] for estimate in operational_estimates]
    df["specialty_margin"] = specialty_margins
    df["specialty_margin_basis"] = specialty_margin_bases
    return df

def rank_model_a(acceptable_df):
    """Model A: 카카오 실제 도로거리가 가까운 순으로 병원을 정렬한다."""
    if acceptable_df.empty:
        return acceptable_df

    df = acceptable_df.copy()
    # Model A의 원래 순위 기준은 카카오 실제 도로거리이다.
    # 화면에는 순서를 보존한 0~100 "추천 점수"로만 표시하며, 순위 로직 자체는 바꾸지 않는다.
    distance_values = pd.to_numeric(df["distance_km"], errors="coerce")
    df["model_score"] = (_normalize_series(distance_values, higher_is_better=False) * 100.0).round(2)
    df["model_score_raw"] = distance_values.round(3)
    df["model_score_label"] = "Model A 거리 기반 추천 점수"

    # 거리 정보가 없는 병원(위경도 미확보)은 맨 뒤로 보낸다.
    df["_sort_distance"] = distance_values.fillna(float("inf"))
    df["_sort_score"] = pd.to_numeric(df["acceptance_score"], errors="coerce").fillna(0)
    # pandas의 기본 정렬(quicksort)은 동률(예: 거리 정보가 없어 전부 inf인 경우)에서
    # 안정 정렬을 보장하지 않아 수용점수 순서가 뒤섞이는 문제가 있었다.
    # 따라서 수용점수를 명시적인 2차 정렬 기준으로 지정해, 거리가 같거나 없을 때는
    # 항상 수용점수가 높은 병원이 먼저 오도록 고정한다.
    df = df.sort_values(
        by=["_sort_distance", "_sort_score"],
        ascending=[True, False],
        kind="mergesort",
    ).drop(columns=["_sort_distance", "_sort_score"])
    df["model_rank_explanation"] = df["distance_km"].apply(
        lambda v: f"구급차 위치로부터 약 {v}km" if v is not None else "거리 정보 없음(병원 좌표 미확보)"
    )
    return df.reset_index(drop=True)



def rank_model_b(acceptable_df, patient=None, weights=None):
    """
    Model B: MILP(혼합정수선형계획) 기반 단일 환자 순위 산정.

    공통 Hard Filter를 통과한 전체 병원을 대상으로, 매 반복마다
      maximize  sum_h x_h * utility_h
      s.t.      sum_h x_h == 1,  x_h ∈ {0,1},  (이미 선택된 병원 제외)
    형태의 0/1 정수계획 문제를 풀어 순위를 만든다.

    utility는 단순 거리가 아니라 예상 치료시작 지연, 수용점수,
    전문의 수, 데이터 신뢰도와 골든타임 초과 패널티를 반영한다.
    다중 환자의 병상 경쟁은 optimize_batch_assignments에서 처리한다.
    """
    if acceptable_df.empty:
        return acceptable_df

    patient = patient or {}
    profile = get_patient_priority_profile(patient)
    if weights is None:
        if profile["name"] == "critical":
            weights = {"delay": 0.45, "score": 0.30, "specialist": 0.15, "reliability": 0.10}
        else:
            weights = {"delay": 0.35, "score": 0.30, "specialist": 0.15, "reliability": 0.20}

    df = acceptable_df.copy()
    df["_norm_score"] = _normalize_series(df["acceptance_score"], higher_is_better=True)
    delay_values = pd.to_numeric(df["effective_treatment_delay_min"], errors="coerce")
    delay_fill = delay_values.max() + 30 if delay_values.notna().any() else 999.0
    df["_norm_delay"] = _normalize_series(delay_values.fillna(delay_fill), higher_is_better=False)
    df["_norm_specialist"] = _normalize_series(
        pd.to_numeric(df["required_specialist_count_total"], errors="coerce").fillna(0),
        higher_is_better=True,
    )
    df["_norm_reliability"] = pd.to_numeric(
        df["data_reliability_score"], errors="coerce"
    ).fillna(0.2).clip(0, 1)

    golden_time = profile["golden_time_min"]
    if golden_time and golden_time > 0:
        df["_golden_violation"] = (
            (delay_values.fillna(delay_fill) - golden_time).clip(lower=0) / golden_time
        ).clip(upper=2.0)
    else:
        df["_golden_violation"] = 0.0

    df["_utility"] = (
        weights["score"] * df["_norm_score"]
        + weights["delay"] * df["_norm_delay"]
        + weights["specialist"] * df["_norm_specialist"]
        + weights["reliability"] * df["_norm_reliability"]
        - 0.25 * profile["severity_weight"] * df["_golden_violation"]
    )

    # Model B의 원래 순위 기준은 MILP utility이다.
    # 화면에는 utility의 순서를 보존한 0~100 "추천 점수"로만 표시한다.
    df["model_score"] = (_normalize_series(df["_utility"], higher_is_better=True) * 100.0).round(2)
    df["model_score_raw"] = df["_utility"].round(6)
    df["model_score_label"] = "Model B MILP 추천 점수"

    remaining_idx = list(df.index)
    ranked_idx = []

    if PULP_AVAILABLE:
        # 후보를 하나씩 제외해가며 0/1 MILP를 반복적으로 풀어 전체 순위를 만든다.
        while remaining_idx:
            prob = pulp.LpProblem("hospital_assignment", pulp.LpMaximize)
            x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in remaining_idx}
            prob += pulp.lpSum(x[i] * float(df.loc[i, "_utility"]) for i in remaining_idx)
            prob += pulp.lpSum(x[i] for i in remaining_idx) == 1
            prob.solve(pulp.PULP_CBC_CMD(msg=False))

            chosen = next((i for i in remaining_idx if pulp.value(x[i]) and pulp.value(x[i]) > 0.5), remaining_idx[0])
            ranked_idx.append(chosen)
            remaining_idx.remove(chosen)
    else:
        # pulp 미설치 시: utility 내림차순 정렬 = 위 MILP를 반복해서 푼 것과 동일한 결과.
        # quicksort는 동률(utility가 같은 경우)에서 안정 정렬을 보장하지 않으므로
        # mergesort(안정 정렬) + 수용점수 2차 기준으로 순서를 고정한다.
        df["_sort_score"] = pd.to_numeric(df["acceptance_score"], errors="coerce").fillna(0)
        ranked_idx = df.sort_values(
            by=["_utility", "_sort_score"],
            ascending=[False, False],
            kind="mergesort",
        ).index.tolist()
        df = df.drop(columns=["_sort_score"])

    df = df.loc[ranked_idx].copy()
    df["model_rank_explanation"] = df.apply(
        lambda r: (
            f"MILP 목적함수 점수 {round(r['_utility'], 3)} "
            f"(예상 치료지연 {r.get('effective_treatment_delay_min', '-')}분, "
            f"{profile['label']} 프로파일)"
        ),
        axis=1,
    )
    df["milp_utility_score"] = df["_utility"].round(6)
    df = df.drop(
        columns=["_norm_score", "_norm_delay", "_norm_specialist", "_norm_reliability", "_golden_violation", "_utility"]
    )
    return df.reset_index(drop=True)



def rank_model_c(acceptable_df, patient=None, variant=DEFAULT_TOPSIS_VARIANT):
    """C0/C1/C2/C3 설정과 공통 엔진을 사용하는 Model C TOPSIS 순위."""
    if acceptable_df.empty:
        return acceptable_df

    patient = patient or {}
    variant = str(variant or DEFAULT_TOPSIS_VARIANT).strip().upper()
    criteria, profile = get_topsis_criteria(variant, patient)
    metadata = get_topsis_variant_metadata(variant, patient)
    df = acceptable_df.copy()
    df["_topsis_closeness"] = _topsis_closeness(df, criteria)
    # Model C의 원래 순위 기준은 TOPSIS closeness이다.
    # 화면에는 같은 순서를 유지하도록 100배한 "추천 점수"로 표시한다.
    df["model_score"] = (df["_topsis_closeness"] * 100.0).round(2)
    df["model_score_raw"] = df["_topsis_closeness"].round(6)
    df["model_score_label"] = f"Model C TOPSIS {variant} 추천 점수"

    # 결정 기준 외의 수용점수가 동률 순위에 숨어들지 않도록
    # 병원명을 공통·중립적 2차 정렬 기준으로 사용한다.
    df["_hospital_name_tiebreaker"] = df.apply(get_hospital_name, axis=1)
    df = df.sort_values(
        by=["_topsis_closeness", "_hospital_name_tiebreaker"],
        ascending=[False, True],
        kind="mergesort",
    )

    profile_text = f", {profile['label']} 가중치" if profile else ""
    experimental_text = ", 실험안" if metadata["experimental"] else ""
    df["model_rank_explanation"] = df["_topsis_closeness"].apply(
        lambda value: (
            f"TOPSIS {variant} 근접도 {round(float(value), 3)} "
            f"({metadata['label']}{profile_text}{experimental_text})"
        )
    )
    df["topsis_closeness"] = df["_topsis_closeness"].round(6)
    df["topsis_variant"] = variant
    df["topsis_variant_label"] = metadata["label"]
    df["topsis_experimental"] = metadata["experimental"]
    df = df.drop(columns=["_topsis_closeness", "_hospital_name_tiebreaker"])
    return df.reset_index(drop=True)



def rank_model_d(acceptable_df, weights=None):
    """
    Model D: 가중치 종합 점수 (Weighted Sum Model).

    수용점수/이동거리/전문의 수/혼잡도를 각각 0~1로 min-max 정규화한 뒤
    사전에 정의한 가중치로 단순 선형 합산한다. TOPSIS(Model C)와 달리
    이상해까지의 거리를 계산하지 않고, "가중치 x 정규화값"을 그대로 더하는
    가장 단순한 다기준 의사결정 방식이다.
    """
    if acceptable_df.empty:
        return acceptable_df

    weights = weights or {"score": 0.45, "distance": 0.25, "specialist": 0.15, "congestion": 0.15}

    df = acceptable_df.copy()
    norm_score = _normalize_series(df["acceptance_score"], higher_is_better=True)
    norm_distance = _normalize_series(df["distance_km"], higher_is_better=False)
    norm_specialist = _normalize_series(df["required_specialist_count_total"], higher_is_better=True)
    norm_congestion = _normalize_series(df["congestion_score"], higher_is_better=False)

    df["_weighted_total"] = (
        weights["score"] * norm_score
        + weights["distance"] * norm_distance
        + weights["specialist"] * norm_specialist
        + weights["congestion"] * norm_congestion
    )
    # Model D의 원래 순위 기준은 개선 ver_0920의 가중합 값이다.
    # 화면에는 같은 순서를 유지하도록 100배한 "추천 점수"로 표시한다.
    df["model_score"] = (df["_weighted_total"] * 100.0).round(2)
    df["model_score_raw"] = df["_weighted_total"].round(6)
    df["model_score_label"] = "Model D 가중합 추천 점수"
    df["weighted_sum_score"] = df["_weighted_total"].round(6)

    df["_sort_score"] = pd.to_numeric(df["acceptance_score"], errors="coerce").fillna(0)
    df = df.sort_values(
        by=["_weighted_total", "_sort_score"],
        ascending=[False, False],
        kind="mergesort",
    )
    df["model_rank_explanation"] = df["_weighted_total"].apply(
        lambda v: f"가중합 종합점수 {round(float(v), 3)} (수용점수 45% · 거리 25% · 전문의수 15% · 혼잡도 15%)"
    )
    df = df.drop(columns=["_weighted_total", "_sort_score"])
    return df.reset_index(drop=True)



def apply_ranking_model(model_key, acceptable_df, patient=None, topsis_variant=DEFAULT_TOPSIS_VARIANT):
    model_key = (model_key or "A").strip().upper()

    if model_key == "B":
        ranked, applied = rank_model_b(acceptable_df, patient=patient), "B"
    elif model_key == "C":
        ranked, applied = rank_model_c(acceptable_df, patient=patient, variant=topsis_variant), "C"
    elif model_key == "D":
        ranked, applied = rank_model_d(acceptable_df), "D"
    else:
        # 기본값 / 잘못된 값이 들어온 경우 Model A로 안전하게 대체
        ranked, applied = rank_model_a(acceptable_df), "A"

    # 표시되는 추천 점수와 표 순서가 어긋나지 않도록 안정 정렬한다.
    # model_score는 각 모델의 원래 순위 기준을 단조 변환한 값이라 기존 모델 순위를 바꾸지 않는다.
    if not ranked.empty and "model_score" in ranked.columns:
        ranked = ranked.sort_values("model_score", ascending=False, kind="mergesort").reset_index(drop=True)
    return ranked, applied


def apply_golden_time_safety_priority(ranked_df, patient=None):
    """
    응급의료 시스템 공통 안전 우선순위.

    A/B/C/D의 내부 계산식과 각 모델이 만든 원래 순서는 그대로 둔 채,
    환자의 golden_time_min 안에 도착 가능한 병원을 먼저 배치한다.

    - ETA <= golden_time: 골든타임 내
    - ETA >  golden_time: 골든타임 초과
    - ETA 미확인: ETA 미확인

    최종 화면의 '추천 점수'와 순위가 다시 어긋나지 않도록,
    원래 모델 점수는 model_score_native에 보존하고 화면용 model_score만
    안전 우선순위를 보존하는 단조 점수로 다시 만든다.
    모델 고유 원시값(model_score_raw / topsis_closeness / milp_utility_score 등)은 변경하지 않는다.
    """
    if ranked_df.empty:
        return ranked_df.copy(), False

    patient = patient or {}
    golden_time = to_number(patient.get("golden_time_min"))
    if golden_time is None or golden_time <= 0 or "eta_min" not in ranked_df.columns:
        out = ranked_df.copy()
        out["golden_time_status"] = "미적용"
        out["golden_time_exceeded"] = False
        return out, False

    out = ranked_df.copy().reset_index(drop=True)
    eta = pd.to_numeric(out["eta_min"], errors="coerce")

    # 모델 자체가 만든 순서를 동일 그룹 안의 2차 기준으로 그대로 보존한다.
    out["_native_model_order"] = np.arange(len(out), dtype=int)
    out["model_score_native"] = pd.to_numeric(out.get("model_score"), errors="coerce")

    within = eta.notna() & (eta <= float(golden_time))
    over = eta.notna() & (eta > float(golden_time))

    out["golden_time_status"] = np.select(
        [within, over],
        ["골든타임 내", "골든타임 초과"],
        default="ETA 미확인",
    )
    out["golden_time_exceeded"] = over

    # 0: 골든타임 내, 1: 초과, 2: ETA 미확인
    # 외부 교통 API가 실패했다고 병원을 탈락시키지는 않되, 확인 가능한 안전 후보를 먼저 보여준다.
    out["_golden_priority"] = np.select([within, over], [0, 1], default=2).astype(int)
    out = out.sort_values(
        by=["_golden_priority", "_native_model_order"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    # 안전 우선순위 적용 후에도 화면 점수가 최종 순서와 일치하도록 표시용 점수만 재매핑한다.
    # 각 안전 그룹 내부에서는 기존 모델 점수의 순서를 그대로 유지한다.
    native_score = pd.to_numeric(out["model_score_native"], errors="coerce")
    native_norm = _normalize_series(native_score.fillna(0), higher_is_better=True)
    priority_bonus = (2 - out["_golden_priority"]) * 2.0
    safety_key = priority_bonus + native_norm
    out["model_score"] = (_normalize_series(safety_key, higher_is_better=True) * 100.0).round(2)

    if "model_score_label" in out.columns:
        out["model_score_label"] = out["model_score_label"].astype(str) + " · 골든타임 우선"

    out = out.drop(columns=["_golden_priority", "_native_model_order"])
    return out, True



# =========================================================
# 5-3. 시스템 성능 검증 지표
# =========================================================

def compute_performance_metrics(
    result_df,
    acceptable_df,
    top_df,
    patient,
    elapsed_ms
):
    total = len(
        result_df
    )

    acceptable_count = len(
        acceptable_df
    )

    acceptance_success_rate = (
        round(
            (
                acceptable_count
                / total
            )
            * 100,
            1
        )
        if total
        else 0.0
    )

    golden_time = to_number(
        patient.get(
            "golden_time_min"
        )
    )

    # ETA 계열 지표는 선택 모델이 실제로 추천한 Top-K를 기준으로 계산한다.
    # 후보 전체를 쓰면 A/B/C/D를 바꿔도 값이 거의 동일해져 모델 비교 지표로 의미가 약해진다.
    if (
        top_df is not None
        and "eta_min" in top_df
    ):
        eta_series = (
            pd.to_numeric(
                top_df["eta_min"],
                errors="coerce"
            ).dropna()
        )
    else:
        eta_series = pd.Series(dtype=float)

    if (
        golden_time
        and not eta_series.empty
    ):
        within_golden = (
            eta_series
            <= golden_time
        ).sum()

        golden_time_compliance_rate = round(
            (
                within_golden
                / len(eta_series)
            )
            * 100,
            1
        )

    else:
        golden_time_compliance_rate = None

    avg_eta = (
        round(
            float(
                eta_series.mean()
            ),
            1
        )
        if not eta_series.empty
        else None
    )

    p95_eta = (
        round(
            float(
                np.percentile(
                    eta_series,
                    95
                )
            ),
            1
        )
        if len(eta_series) > 0
        else None
    )

    if (
        "congestion_score"
        in top_df
    ):
        congestion_series = (
            pd.to_numeric(
                top_df[
                    "congestion_score"
                ],
                errors="coerce"
            ).dropna()
        )

    else:
        congestion_series = (
            pd.Series(
                dtype=float
            )
        )

    avg_congestion_pct = (
        round(
            float(
                congestion_series.mean()
            )
            * 100,
            1
        )
        if not congestion_series.empty
        else None
    )

    return {
        "acceptance_success_rate_pct": (
            acceptance_success_rate
        ),
        "golden_time_compliance_rate_pct": (
            golden_time_compliance_rate
        ),
        "avg_eta_min": avg_eta,
        "p95_eta_min": p95_eta,
        "hospital_congestion_pct": (
            avg_congestion_pct
        ),
        "computation_time_ms": round(
            elapsed_ms,
            1
        ),
        "excluded_metrics": [
            {
                "name": "재매칭률",
                "reason": (
                    "병원의 실제 거절/재요청 이력을 "
                    "받는 연동이 아직 없어 항상 "
                    "0으로만 계산되어 의미가 없습니다."
                ),
            },
            {
                "name": "도착 시 수용률",
                "reason": (
                    "구급차 도착 시점의 병원 최종 "
                    "확정 응답을 받는 연동이 아직 없어 "
                    "계산할 수 없습니다."
                ),
            },
        ],
    }


# =========================================================
# 6. 환자 카테고리별 요구 조건 보정
# =========================================================

def enrich_patient_by_category(
    patient
):
    patient = dict(patient)

    category = str(
        patient.get(
            "suspected_category",
            ""
        )
    ).strip()

    equipment = list(
        patient.get(
            "equipment",
            []
        )
        or []
    )

    specialists = list(
        patient.get(
            "specialists",
            []
        )
        or []
    )

    def add_equipment(name):
        if name not in equipment:
            equipment.append(name)

    def add_specialist(name):
        if name not in specialists:
            specialists.append(name)

    patient[
        "req_er_bed"
    ] = bool(
        patient.get(
            "req_er_bed",
            True
        )
    )

    if any(
        keyword in category
        for keyword in [
            "뇌졸중",
            "뇌출혈",
            "신경",
            "중풍",
        ]
    ):
        add_equipment("CT")
        add_equipment("MRI")

        add_specialist(
            "신경과"
        )

        add_specialist(
            "신경외과"
        )

        patient[
            "req_icu"
        ] = True

        patient[
            "req_severe_acceptance"
        ] = True

    elif any(
        keyword in category
        for keyword in [
            "심근경색",
            "심장",
            "심혈관",
            "흉통",
        ]
    ):
        add_equipment(
            "CAG"
        )

        add_equipment(
            "혈관조영"
        )

        add_specialist(
            "내과"
        )

        add_specialist(
            "심장혈관흉부외과"
        )

        patient[
            "req_icu"
        ] = True

        patient[
            "req_severe_acceptance"
        ] = True

    elif any(
        keyword in category
        for keyword in [
            "외상",
            "교통사고",
            "다발성",
            "압궤",
            "골절",
        ]
    ):
        add_equipment(
            "CT"
        )

        add_specialist(
            "외과"
        )

        add_specialist(
            "정형외과"
        )

        patient[
            "req_icu"
        ] = True

        patient[
            "req_or"
        ] = True

        patient[
            "req_severe_acceptance"
        ] = True

    elif any(
        keyword in category
        for keyword in [
            "산모",
            "임신",
            "분만",
            "전자간증",
            "임신중독",
        ]
    ):
        add_specialist(
            "산부인과"
        )

        add_specialist(
            "소아청소년과"
        )

        patient[
            "req_or"
        ] = True

        patient[
            "req_icu"
        ] = True

        patient[
            "req_severe_acceptance"
        ] = True

    elif any(
        keyword in category
        for keyword in [
            "소아",
            "신생아",
            "영아",
        ]
    ):
        add_specialist(
            "소아청소년과"
        )

        add_specialist(
            "응급의학과"
        )

        patient[
            "req_er_bed"
        ] = True

        patient[
            "req_severe_acceptance"
        ] = True

    elif any(
        keyword in category
        for keyword in [
            "호흡곤란",
            "감염",
            "패혈증",
            "폐렴",
        ]
    ):
        add_equipment(
            "인공호흡기"
        )

        add_specialist(
            "응급의학과"
        )

        patient[
            "req_icu"
        ] = True

        patient[
            "req_severe_acceptance"
        ] = True

    patient[
        "equipment"
    ] = equipment

    patient[
        "specialists"
    ] = specialists

    return patient


# =========================================================
# 7. 조건별 판별 함수
# =========================================================

def check_er_bed(
    hospital,
    patient
):
    if not patient.get(
        "req_er_bed",
        True
    ):
        return True

    return numeric_available(
        hospital.get("hvec")
    )


def check_or(
    hospital,
    patient
):
    if not patient.get(
        "req_or",
        False
    ):
        return True

    return numeric_available(
        hospital.get("hvoc")
    )


def check_equipment(
    hospital,
    patient
):
    """
    실시간 응급의료 API 장비 가능 여부를 1순위로 사용한다.
    실시간 값이 비어 있을 때만
    HIRA medical_equipment_summary를 보조 근거로 사용한다.
    """

    required_equipment = (
        patient.get(
            "equipment",
            []
        )
        or []
    )

    equipment_map = {
        "CT": {
            "col": "hvctayn",
            "keywords": [
                "CT",
                "전산화단층",
                "전산화 단층",
            ],
        },
        "MRI": {
            "col": "hvmriayn",
            "keywords": [
                "MRI",
                "자기공명",
            ],
        },
        "CAG": {
            "col": "hvangioayn",
            "keywords": [
                "혈관조영",
                "ANGIO",
                "CAG",
                "심혈관조영",
            ],
        },
        "혈관조영": {
            "col": "hvangioayn",
            "keywords": [
                "혈관조영",
                "ANGIO",
                "CAG",
                "심혈관조영",
            ],
        },
        "조영": {
            "col": "hvangioayn",
            "keywords": [
                "혈관조영",
                "ANGIO",
                "CAG",
                "심혈관조영",
            ],
        },
        "인공호흡기": {
            "col": "hvventiayn",
            "keywords": [
                "인공호흡",
                "VENTILATOR",
                "호흡기",
            ],
        },
        "VENTILATOR": {
            "col": "hvventiayn",
            "keywords": [
                "인공호흡",
                "VENTILATOR",
                "호흡기",
            ],
        },
        "호흡기": {
            "col": "hvventiayn",
            "keywords": [
                "인공호흡",
                "VENTILATOR",
                "호흡기",
            ],
        },
    }

    equipment_summary = str(
        hospital.get(
            "medical_equipment_summary",
            ""
        )
    )

    passed = []
    missing = []
    unknown = []
    static_supported = []

    for equip in required_equipment:
        equip_name = str(
            equip
        ).strip()

        equip_upper = (
            equip_name.upper()
        )

        rule = None

        for (
            key,
            candidate,
        ) in equipment_map.items():

            if (
                key.upper()
                in equip_upper
                or key in equip_name
            ):
                rule = candidate
                break

        if rule is None:
            unknown.append(
                equip_name
            )
            continue

        realtime_value = (
            yn_to_bool(
                hospital.get(
                    rule["col"]
                )
            )
        )

        if realtime_value is True:
            passed.append(
                equip_name
            )

        elif realtime_value is False:
            missing.append(
                equip_name
            )

        else:
            if text_contains_any(
                equipment_summary,
                rule["keywords"]
            ):
                static_supported.append(
                    equip_name
                )

                unknown.append(
                    f"{equip_name}"
                    "(장비 보유 확인, "
                    "실시간 가용 확인 필요)"
                )

            else:
                unknown.append(
                    equip_name
                )

    equipment_ok = (
        len(missing) == 0
    )

    return (
        equipment_ok,
        passed,
        missing,
        unknown,
        static_supported,
    )


def check_icu(
    hospital,
    patient
):
    if not patient.get(
        "req_icu",
        False
    ):
        return True, [], []

    category = str(
        patient.get(
            "suspected_category",
            ""
        )
    )

    icu_cols = [
        "hvicc"
    ]

    if any(
        keyword in category
        for keyword in [
            "뇌",
            "신경",
            "뇌졸중",
            "뇌출혈",
        ]
    ):
        icu_cols = [
            "hvcc",
            "hv6",
            "hvicc",
        ]

    elif any(
        keyword in category
        for keyword in [
            "소아",
            "신생아",
            "영아",
        ]
    ):
        icu_cols = [
            "hvncc",
            "hvicc",
        ]

    elif any(
        keyword in category
        for keyword in [
            "심근경색",
            "심장",
            "심혈관",
        ]
    ):
        icu_cols = [
            "hvccc",
            "hvicc",
        ]

    elif any(
        keyword in category
        for keyword in [
            "외상",
            "교통사고",
            "다발성",
            "압궤",
        ]
    ):
        icu_cols = [
            "hv3",
            "hvicc",
        ]

    unknown_cols = []

    for col in icu_cols:
        available = (
            numeric_available(
                hospital.get(col)
            )
        )

        if available is True:
            return (
                True,
                icu_cols,
                unknown_cols
            )

        if available is None:
            unknown_cols.append(
                col
            )

    if unknown_cols:
        return (
            None,
            icu_cols,
            unknown_cols
        )

    return (
        False,
        icu_cols,
        unknown_cols
    )


def specialist_rules_for_requirement(
    req
):
    """
    환자에게 필요한 진료과명을
    HIRA 과별 전문의 수 컬럼명과 매칭하기 위한 규칙.
    """

    req = str(req).strip()

    rules = {
        "내과": {
            "flag": None,
            "keywords": [
                "내과"
            ],
        },
        "응급의학과": {
            "flag": (
                "has_emergency_medicine"
            ),
            "keywords": [
                "응급의학과"
            ],
        },
        "신경과": {
            "flag": (
                "has_neurology"
            ),
            "keywords": [
                "신경과"
            ],
        },
        "신경외과": {
            "flag": (
                "has_neurosurgery"
            ),
            "keywords": [
                "신경외과"
            ],
        },
        "심장내과": {
            "flag": None,
            "keywords": [
                "내과",
                "심장혈관흉부외과",
                "흉부외과",
                "심혈관",
            ],
        },
        "순환기내과": {
            "flag": None,
            "keywords": [
                "내과",
                "심장혈관흉부외과",
                "흉부외과",
                "심혈관",
            ],
        },
        "외과": {
            "flag": (
                "has_general_surgery"
            ),
            "keywords": [
                "외과"
            ],
        },
        "흉부외과": {
            "flag": (
                "has_thoracic_surgery"
            ),
            "keywords": [
                "흉부외과",
                "심장혈관흉부외과",
            ],
        },
        "정형외과": {
            "flag": (
                "has_orthopedics"
            ),
            "keywords": [
                "정형외과"
            ],
        },
        "산부인과": {
            "flag": "has_obgyn",
            "keywords": [
                "산부인과"
            ],
        },
        "소아청소년과": {
            "flag": (
                "has_pediatrics"
            ),
            "keywords": [
                "소아청소년과",
                "소아과",
            ],
        },
        "소아과": {
            "flag": (
                "has_pediatrics"
            ),
            "keywords": [
                "소아청소년과",
                "소아과",
            ],
        },
    }

    for (
        key,
        rule,
    ) in rules.items():

        if (
            key in req
            or req in key
        ):
            return rule

    return {
        "flag": None,
        "keywords": [
            req
        ],
    }


def get_specialist_count_for_rule(
    hospital,
    rule
):
    """
    병원 DB의 모든
    specialist_count_코드_진료과명 컬럼을 훑어서
    필요한 진료과 키워드와 맞는 전문의 수를 합산한다.
    """

    if not rule:
        return None

    import re

    total = 0
    matched = False

    for (
        col,
        raw_value,
    ) in hospital.items():

        col_name = str(col)

        if not re.match(
            r"^specialist_count_\d+_",
            col_name
        ):
            continue

        if any(
            keyword in col_name
            for keyword in rule.get(
                "keywords",
                []
            )
        ):
            value = to_number(
                raw_value
            )

            if (
                value is not None
                and value > 0
            ):
                total += value
                matched = True

    if not matched:
        return None

    return int(total)


def check_specialists(
    hospital,
    patient
):
    specialists = (
        patient.get(
            "specialists",
            []
        )
        or []
    )

    if not specialists:
        return (
            True,
            [],
            [],
            [],
            0,
            "",
        )

    departments_text = str(
        hospital.get(
            "departments",
            ""
        )
    ).strip()

    passed = []
    missing = []
    unknown = []
    count_parts = []

    total_required_specialist_count = 0

    for req in specialists:
        req = str(req).strip()

        if not req:
            continue

        rule = (
            specialist_rules_for_requirement(
                req
            )
        )

        if rule is None:
            if (
                departments_text
                and req
                in departments_text
            ):
                passed.append(
                    req
                )

            elif departments_text:
                unknown.append(
                    req
                )

            else:
                unknown.append(
                    req
                )

            continue

        flag_col = (
            rule.get("flag")
        )

        flag_value = (
            normalize_yes_no(
                hospital.get(
                    flag_col
                )
            )
            if flag_col
            else None
        )

        count_value = (
            get_specialist_count_for_rule(
                hospital,
                rule
            )
        )

        if (
            count_value
            is not None
        ):
            total_required_specialist_count += (
                count_value
            )

            count_parts.append(
                f"{req}:"
                f"{count_value}명"
            )

        text_match = (
            departments_text
            and any(
                keyword
                in departments_text
                for keyword
                in rule[
                    "keywords"
                ]
            )
        )

        if (
            flag_value is True
            or text_match
            or (
                count_value
                is not None
                and count_value > 0
            )
        ):
            passed.append(
                req
            )

        elif flag_value is False:
            missing.append(
                req
            )

        else:
            unknown.append(
                req
            )

    if missing:
        status = False

    elif unknown:
        status = None

    else:
        status = True

    return (
        status,
        passed,
        missing,
        unknown,
        total_required_specialist_count,
        ", ".join(
            count_parts
        ),
    )


def check_special_diag(
    hospital,
    patient
):
    """
    HIRA 특수진료정보는 정적 병원 역량 보조 지표로 사용한다.
    없다고 탈락시키지는 않고 점수 보정 및 사유 표시만 한다.
    """

    category = str(
        patient.get(
            "suspected_category",
            ""
        )
    )

    summary = str(
        hospital.get(
            "special_diag_summary",
            ""
        )
    )

    if not summary:
        return 0, ""

    keyword_sets = []

    if any(
        k in category
        for k in [
            "뇌졸중",
            "뇌출혈",
            "신경",
        ]
    ):
        keyword_sets = [
            "뇌",
            "신경",
            "중환자",
            "응급",
        ]

    elif any(
        k in category
        for k in [
            "심근경색",
            "심장",
            "심혈관",
            "흉통",
        ]
    ):
        keyword_sets = [
            "심장",
            "심혈관",
            "혈관",
            "중환자",
            "응급",
        ]

    elif any(
        k in category
        for k in [
            "외상",
            "교통사고",
            "다발성",
            "압궤",
        ]
    ):
        keyword_sets = [
            "외상",
            "수술",
            "중환자",
            "응급",
        ]

    elif any(
        k in category
        for k in [
            "산모",
            "임신",
            "분만",
            "전자간증",
        ]
    ):
        keyword_sets = [
            "분만",
            "산부",
            "신생아",
            "중환자",
        ]

    elif any(
        k in category
        for k in [
            "소아",
            "신생아",
            "영아",
        ]
    ):
        keyword_sets = [
            "소아",
            "신생아",
            "응급",
            "중환자",
        ]

    elif any(
        k in category
        for k in [
            "호흡곤란",
            "감염",
            "패혈증",
            "폐렴",
        ]
    ):
        keyword_sets = [
            "호흡",
            "중환자",
            "응급",
            "감염",
        ]

    if not keyword_sets:
        return 0, ""

    matched = [
        keyword
        for keyword
        in keyword_sets
        if keyword in summary
    ]

    if not matched:
        return 0, ""

    bonus = min(
        5,
        len(matched) * 2
    )

    return (
        bonus,
        (
            "특수진료정보 관련 키워드 확인: "
            + ", ".join(
                matched
            )
        ),
    )


def specialist_count_score(
    total_required_specialist_count
):
    """
    필수 진료과가 있는 병원 중
    전문의 수가 많을수록 가산한다.
    """

    if (
        total_required_specialist_count
        is None
        or total_required_specialist_count
        <= 0
    ):
        return 0

    if (
        total_required_specialist_count
        >= 20
    ):
        return 10

    if (
        total_required_specialist_count
        >= 10
    ):
        return 8

    if (
        total_required_specialist_count
        >= 5
    ):
        return 6

    if (
        total_required_specialist_count
        >= 2
    ):
        return 4

    return 2


def check_severe_acceptance(
    hospital,
    patient
):
    if not patient.get(
        "req_severe_acceptance",
        False
    ):
        return True

    er_ok = check_er_bed(
        hospital,
        patient
    )

    (
        icu_ok,
        _,
        _,
    ) = check_icu(
        hospital,
        patient
    )

    if (
        er_ok is False
        or icu_ok is False
    ):
        return False

    if (
        er_ok is None
        or icu_ok is None
    ):
        return None

    return True


# =========================================================
# 8. 병원별 최종 수용 가능 여부 판별
# =========================================================

def evaluate_hospital_acceptance(
    hospital,
    patient
):
    passed_reasons = []
    failed_reasons = []
    unknown_reasons = []

    realtime_status = str(
        hospital.get(
            "realtime_match_status",
            ""
        )
    ).strip()

    if (
        realtime_status
        and realtime_status != "OK"
    ):
        unknown_reasons.append(
            "응급의료기관 실시간 API "
            "미매칭/정보없음"
        )

    er_ok = check_er_bed(
        hospital,
        patient
    )

    if er_ok is True:
        passed_reasons.append(
            "응급실 가용병상 조건 충족"
        )

    elif er_ok is False:
        failed_reasons.append(
            "응급실 가용병상 없음"
        )

    else:
        unknown_reasons.append(
            "응급실 가용병상 확인불가"
        )

    (
        equip_ok,
        passed_equips,
        missing_equips,
        unknown_equips,
        static_supported_equips,
    ) = check_equipment(
        hospital,
        patient
    )

    if passed_equips:
        passed_reasons.append(
            "실시간 필수 장비 가능: "
            + ", ".join(
                passed_equips
            )
        )

    if static_supported_equips:
        passed_reasons.append(
            "HIRA 장비 보유 정보 확인: "
            + ", ".join(
                static_supported_equips
            )
        )

    if missing_equips:
        failed_reasons.append(
            "필수 장비 부족: "
            + ", ".join(
                missing_equips
            )
        )

    if unknown_equips:
        unknown_reasons.append(
            "장비 실시간 가용 확인필요: "
            + ", ".join(
                unknown_equips
            )
        )

    (
        icu_ok,
        icu_cols,
        unknown_icu_cols,
    ) = check_icu(
        hospital,
        patient
    )

    if icu_ok is True:
        passed_reasons.append(
            "ICU 조건 충족"
        )

    elif icu_ok is False:
        failed_reasons.append(
            "ICU 조건 미충족"
        )

    else:
        unknown_reasons.append(
            "ICU 가용 여부 확인불가: "
            + ", ".join(
                unknown_icu_cols
            )
        )

    or_ok = check_or(
        hospital,
        patient
    )

    if or_ok is True:
        passed_reasons.append(
            "수술실 조건 충족"
        )

    elif or_ok is False:
        failed_reasons.append(
            "수술실 가용 불가"
        )

    else:
        unknown_reasons.append(
            "수술실 가용 여부 확인불가"
        )

    severe_ok = (
        check_severe_acceptance(
            hospital,
            patient
        )
    )

    if severe_ok is True:
        passed_reasons.append(
            "중증환자 수용 조건 충족"
        )

    elif severe_ok is False:
        failed_reasons.append(
            "중증환자 수용 조건 미충족"
        )

    else:
        unknown_reasons.append(
            "중증환자 수용 가능 여부 "
            "확인불가"
        )

    (
        specialist_ok,
        passed_specialists,
        missing_specialists,
        unknown_specialists,
        required_specialist_count_total,
        required_specialist_count_text,
    ) = check_specialists(
        hospital,
        patient
    )

    if specialist_ok is True:
        if passed_specialists:
            passed_reasons.append(
                "필수 진료과 조건 충족: "
                + ", ".join(
                    passed_specialists
                )
            )

    elif specialist_ok is False:
        failed_reasons.append(
            "필수 진료과 없음: "
            + ", ".join(
                missing_specialists
            )
        )

    else:
        if passed_specialists:
            passed_reasons.append(
                "확인된 진료과: "
                + ", ".join(
                    passed_specialists
                )
            )

        if unknown_specialists:
            unknown_reasons.append(
                "필수 진료과 확인불가: "
                + ", ".join(
                    unknown_specialists
                )
            )

    if required_specialist_count_text:
        passed_reasons.append(
            "필수 진료과 전문의 수: "
            + required_specialist_count_text
        )

    (
        special_diag_bonus,
        special_diag_reason,
    ) = check_special_diag(
        hospital,
        patient
    )

    if special_diag_reason:
        passed_reasons.append(
            special_diag_reason
        )

    accept_possible = (
        len(failed_reasons) == 0
    )

    if (
        accept_possible
        and unknown_reasons
    ):
        acceptance_status = (
            "수용 가능 후보_일부 확인필요"
        )

    elif accept_possible:
        acceptance_status = (
            "수용 가능"
        )

    else:
        acceptance_status = (
            "수용 불가"
        )

    score = 0

    if er_ok is True:
        score += 18

    if equip_ok:
        score += 17

    if icu_ok is True:
        score += 18

    if or_ok is True:
        score += 8

    if severe_ok is True:
        score += 9

    if specialist_ok is True:
        score += 20

    elif (
        specialist_ok is None
        and passed_specialists
    ):
        score += 10

    elif specialist_ok is False:
        score = max(
            0,
            score - 30
        )

    score += specialist_count_score(
        required_specialist_count_total
    )

    score += special_diag_bonus

    if static_supported_equips:
        score += min(
            3,
            len(
                static_supported_equips
            )
        )

    if realtime_status == "OK":
        score += 2

    if failed_reasons:
        score = max(
            0,
            score - 40
        )

    score = min(
        score,
        100
    )

    return {
        "accept_possible": (
            accept_possible
        ),
        "acceptance_status": (
            acceptance_status
        ),
        "acceptance_score": (
            int(score)
        ),
        "checked_icu_columns": (
            icu_cols
        ),
        "required_specialists_text": (
            ", ".join(
                patient.get(
                    "specialists",
                    []
                )
                or []
            )
        ),
        "required_specialists_passed_text": (
            ", ".join(
                passed_specialists
            )
        ),
        "required_specialists_missing_text": (
            ", ".join(
                missing_specialists
            )
        ),
        "required_specialists_unknown_text": (
            ", ".join(
                unknown_specialists
            )
        ),
        "required_specialist_count_total": (
            int(
                required_specialist_count_total
                or 0
            )
        ),
        "required_specialist_count_text": (
            required_specialist_count_text
        ),
        "equipment_static_supported_text": (
            ", ".join(
                static_supported_equips
            )
        ),
        "special_diag_bonus": (
            int(
                special_diag_bonus
            )
        ),
        "passed_reasons": (
            passed_reasons
        ),
        "failed_reasons": (
            failed_reasons
        ),
        "unknown_reasons": (
            unknown_reasons
        ),
        "passed_reasons_text": (
            " / ".join(
                passed_reasons
            )
        ),
        "failed_reasons_text": (
            " / ".join(
                failed_reasons
            )
        ),
        "unknown_reasons_text": (
            " / ".join(
                unknown_reasons
            )
        ),
    }


def filter_acceptable_hospitals(
    hospital_df,
    patient
):
    results = []

    for (
        _,
        hospital,
    ) in hospital_df.iterrows():

        row = hospital.to_dict()

        evaluation = (
            evaluate_hospital_acceptance(
                row,
                patient
            )
        )

        row.update(
            evaluation
        )

        row[
            "hospital_display_name"
        ] = get_hospital_name(
            row
        )

        results.append(
            row
        )

    result_df = pd.DataFrame(
        results
    )

    if result_df.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    result_df = (
        result_df.sort_values(
            by=[
                "accept_possible",
                "acceptance_score",
                "required_specialist_count_total",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            kind="mergesort",
        ).reset_index(
            drop=True
        )
    )

    acceptable_df = result_df[
        result_df[
            "accept_possible"
        ] == True
    ].copy()

    rejected_df = result_df[
        result_df[
            "accept_possible"
        ] == False
    ].copy()

    return (
        acceptable_df,
        rejected_df,
        result_df,
    )


# =========================================================
# 9. 환자 분석 + 병원 수용 가능 여부 판별 API
# =========================================================

@app.route(
    "/api/recommend-hospitals",
    methods=["POST"]
)
def recommend_hospitals_api():
    start_time = time.perf_counter()

    try:
        data = request.get_json(silent=True) or {}
        supplied_patient = data.get("patient")
        text = str(data.get("text") or "").strip()

        if supplied_patient:
            # A/B/C/D 비교 시 첫 Groq PatientInfo 결과를 그대로 재사용한다.
            raw_patient = PatientInfo.model_validate(supplied_patient).model_dump()
            patient_analysis_reused = True
        else:
            if not text:
                return jsonify({"success": False, "error": "text 또는 patient가 필요합니다."}), 400
            cached_patient = get_cached_patient_analysis(text)
            if cached_patient is not None:
                raw_patient = cached_patient
                patient_analysis_reused = True
            else:
                raw_patient = analyze_patient(text)
                patient_analysis_reused = False

        patient = enrich_patient_by_category(raw_patient)
        ambulance_lat = to_number(data.get("ambulance_lat")) or DEFAULT_AMBULANCE_LAT
        ambulance_lng = to_number(data.get("ambulance_lng")) or DEFAULT_AMBULANCE_LNG
        model_key = data.get("model", "A")
        topsis_variant = data.get("topsis_variant", DEFAULT_TOPSIS_VARIANT)

        hospital_df, excel_file, sheet_name = load_hospital_db()
        acceptable_all_df, rejected_df, result_df = filter_acceptable_hospitals(hospital_df, patient)

        # Hard Filter 통과 수는 전체 후보 기준으로 유지한다.
        # 미확인(U)은 기존 로직대로 수용 가능 후보에 남고,
        # 명확한 불충족(N)만 rejected_df(수용 불가)로 분리된다.
        hard_filter_acceptable_count = len(acceptable_all_df)

        # 추천 후보의 라우팅은 Kakao 다중 목적지 최대 1~2 batch로 제한한다.
        # Kakao에서 특정 병원을 받지 못해도 그 병원을 제거하지 않고 추정값으로 유지한다.
        model_candidate_df, kakao_confirmed_count, routing_batch_count = (
            _prepare_routing_candidates_two_batches(
                acceptable_all_df,
                ambulance_lat,
                ambulance_lng,
                target_confirmed=KAKAO_TARGET_CONFIRMED_ROUTES,
            )
        )

        if model_candidate_df.empty and not acceptable_all_df.empty:
            # Hard Filter는 통과했지만 병원 좌표가 전부 비어 있는 경우에도
            # 의료적으로 수용 가능한 병원을 API 오류처럼 숨기지 않는다.
            # 거리/ETA는 미확인 상태로 두고 기존 모델의 나머지 기준으로 순위를 계산한다.
            model_candidate_df = acceptable_all_df.copy()
            model_candidate_df["distance_km"] = np.nan
            model_candidate_df["eta_min"] = np.nan
            model_candidate_df["kakao_route_ok"] = False
            model_candidate_df["routing_source"] = "coordinate_unavailable"

        model_candidate_df = attach_model_feature_fields(model_candidate_df, patient=patient)

        ranked_acceptable_df, applied_model_key = apply_ranking_model(
            model_key,
            model_candidate_df,
            patient=patient,
            topsis_variant=topsis_variant,
        )

        # A/B/C/D의 내부 로직은 그대로 두고, 시스템 공통 안전장치로
        # 골든타임 내 도착 가능한 병원을 우선 배치한다.
        ranked_acceptable_df, golden_time_safety_applied = apply_golden_time_safety_priority(
            ranked_acceptable_df,
            patient=patient,
        )

        # 추천 목록은 수용 가능 후보만 최대 10개 표시한다.
        # 10개가 부족하더라도 명확한 불충족 병원을 숫자 채우기용으로 올리지 않는다.
        top_acceptable_df = ranked_acceptable_df.head(10).copy()
        top_result_df = top_acceptable_df.copy()

        # 수용 불가 병원은 추천 10개에 섞지 않고 기존 '수용 불가' 탭에서 별도로 보여준다.
        # 여기서는 외부 Kakao API를 추가 호출하지 않는다.
        rejected_display_df = rejected_df.copy()

        # 전체 Hard Filter 통과 집합에 ETA 컬럼을 만들어 성능지표의 분모는 전체 통과 수로 유지한다.
        # ETA는 Kakao가 확보된 후보는 실제값, 미확보 후보는 명시적인 추정값을 사용한다.
        acceptable_metrics_df = acceptable_all_df.copy()
        acceptable_metrics_df["eta_min"] = np.nan
        if not model_candidate_df.empty:
            acceptable_metrics_df.loc[model_candidate_df.index, "eta_min"] = model_candidate_df["eta_min"]

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        performance_metrics = compute_performance_metrics(
            result_df=result_df,
            acceptable_df=acceptable_metrics_df,
            top_df=top_acceptable_df,
            patient=patient,
            elapsed_ms=elapsed_ms,
        )

        with RUNTIME_DB_LOCK:
            runtime_meta = dict(RUNTIME_HOSPITAL_META)

        response_payload = {
            "success": True,
            "source_file": excel_file,
            "sheet_name": sheet_name,
            "total_hospital_count": len(result_df),
            "display_hospital_count": len(top_acceptable_df),
            "patient": patient,
            "patient_analysis_reused": patient_analysis_reused,
            "acceptable_count": hard_filter_acceptable_count,
            "rejected_count": len(rejected_df),
            "model_candidate_count": len(model_candidate_df),
            "kakao_confirmed_count": int(kakao_confirmed_count),
            "estimated_route_count": int(
                (model_candidate_df.get("routing_source", pd.Series(dtype=str)) == "estimated_fallback").sum()
            ),
            "routing_batch_count": int(routing_batch_count),
            "golden_time_safety_applied": bool(golden_time_safety_applied),
            "golden_time_min": to_number(patient.get("golden_time_min")),
            "display_acceptable_count": len(top_acceptable_df),
            "display_rejected_count": len(rejected_display_df),
            "acceptable_hospitals": dataframe_to_records(top_acceptable_df),
            "rejected_hospitals": dataframe_to_records(rejected_display_df),
            "all_results": dataframe_to_records(top_result_df),
            "applied_model": {**MODEL_INFO[applied_model_key], "key": applied_model_key},
            "ambulance_location": {"lat": ambulance_lat, "lng": ambulance_lng},
            "performance_metrics": performance_metrics,
            "routing_source": "kakao_multi_with_estimated_fallback",
            "db_updated_at": runtime_meta.get("db_updated_at"),
            "db_synced_at": runtime_meta.get("synced_at"),
        }
        if applied_model_key == "C":
            response_payload["topsis"] = get_topsis_variant_metadata(topsis_variant, patient)

        return jsonify(response_payload)

    except Exception as e:
        print("RECOMMEND ERROR:", e)
        status_code = _groq_status_code(e)
        if status_code == 429:
            return jsonify({
                "success": False,
                "error": "Groq 무료 API 요청 한도에 도달했습니다. 같은 환자 입력이 이미 분석된 경우에는 저장된 PatientInfo를 재사용해 A/B/C/D 비교를 계속할 수 있습니다.",
            }), 429
        if status_code is not None and status_code >= 500:
            return jsonify({
                "success": False,
                "error": "Groq 모델 서버에서 일시적인 오류가 발생했습니다. 같은 환자 입력이 이미 분석된 경우에는 저장된 PatientInfo를 재사용해 A/B/C/D 비교를 계속할 수 있습니다.",
            }), 503
        return jsonify({"success": False, "error": str(e)}), 500



# =========================================================
# 9-1. 병원 사전 수용 신청 / 환자 정보 전송
# =========================================================

@app.route(
    "/api/dispatch-to-hospital",
    methods=["POST"]
)
def dispatch_to_hospital_api():
    """
    사용자가 병원 순위 대시보드에서 병원을 선택하고
    '병원 사전 수용 신청 및 이송 시작' 버튼을 눌렀을 때 호출된다.

    실제 병원 EMR/연계 시스템 API가 아직 없으므로,
    전송할 환자 정보(payload)를 검증하고
    dispatch_log.jsonl에 기록하는 것으로
    "병원측 전송"을 시뮬레이션한다.
    """

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": (
                    "요청 본문이 비어 있습니다."
                )
            }), 400

        hospital = data.get(
            "hospital"
        )

        patient = data.get(
            "patient"
        )

        if (
            not hospital
            or not patient
        ):
            return jsonify({
                "success": False,
                "error": (
                    "hospital, patient 정보가 "
                    "모두 필요합니다."
                )
            }), 400

        dispatch_record = {
            "dispatch_id": (
                "DP-"
                + datetime.now().strftime(
                    "%Y%m%d%H%M%S%f"
                )
            ),
            "dispatched_at": (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ),
            "hospital_name": (
                hospital.get(
                    "hospital_display_name"
                )
                or hospital.get(
                    "target_hospital"
                )
                or hospital.get(
                    "hospital_name"
                )
            ),
            "hospital_raw": hospital,
            "patient": patient,
            "eta_min": hospital.get(
                "eta_min"
            ),
            "distance_km": hospital.get(
                "distance_km"
            ),
            "applied_model": data.get(
                "applied_model"
            ),
        }

        with open(
            DISPATCH_LOG_FILE,
            "a",
            encoding="utf-8"
        ) as f:
            f.write(
                json.dumps(
                    dispatch_record,
                    ensure_ascii=False,
                    default=str
                )
                + "\n"
            )

        print(
            f"[DISPATCH] "
            f"{dispatch_record['hospital_name']} "
            "로 환자 정보 전송 완료 "
            f"(dispatch_id="
            f"{dispatch_record['dispatch_id']})"
        )

        return jsonify(
            {
                "success": True,
                "dispatch_id": (
                    dispatch_record[
                        "dispatch_id"
                    ]
                ),
                "message": (
                    f"{dispatch_record['hospital_name']}"
                    "에 환자 정보를 전송했습니다."
                ),
            }
        )

    except Exception as e:
        print(
            "DISPATCH ERROR:",
            e
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# 10. 화면 파일 제공
# =========================================================

@app.route(
    "/api/random-patient-location",
    methods=["POST", "GET"]
)
def random_patient_location_api():
    """
    SAVER 화면 안에서 환자 위치를 랜덤 생성하기 위한 API.
    전국 병원 DB 중 좌표가 있는 병원을 하나 고른 뒤,
    해당 병원 주변 0.8~5km 지점에 환자 위치를 생성한다.
    """

    try:
        (
            df,
            excel_file,
            sheet_name,
        ) = load_hospital_db()

        candidate_rows = []

        for (
            _,
            row,
        ) in df.iterrows():

            row_dict = (
                row.to_dict()
            )

            (
                lat,
                lng,
            ) = get_hospital_lat_lng(
                row_dict
            )

            if (
                lat is not None
                and lng is not None
                and is_valid_korean_coordinate(
                    lat,
                    lng
                )
            ):
                candidate_rows.append(
                    row_dict
                )

        if not candidate_rows:
            return jsonify({
                "success": False,
                "error": (
                    "좌표가 있는 병원을 "
                    "찾지 못했습니다."
                )
            }), 400

        selected = random.choice(
            candidate_rows
        )

        location = (
            random_location_near_hospital(
                selected
            )
        )

        if not location:
            return jsonify({
                "success": False,
                "error": (
                    "환자 위치 생성에 실패했습니다."
                )
            }), 500

        return jsonify({
            "success": True,
            "source_file": (
                excel_file
            ),
            "sheet_name": (
                sheet_name
            ),
            "patient_location": (
                location
            ),
        })

    except Exception as e:
        print(
            "RANDOM LOCATION ERROR:",
            e
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route(
    "/api/route",
    methods=["GET"]
)
def kakao_route_api():
    """
    지도팀의 /api/route를
    Flask SAVER 백엔드로 통합한 API.
    """

    try:
        origin_lat = request.args.get(
            "originLat"
        )

        origin_lng = request.args.get(
            "originLng"
        )

        destination_lat = (
            request.args.get(
                "destinationLat"
            )
        )

        destination_lng = (
            request.args.get(
                "destinationLng"
            )
        )

        if not all([
            origin_lat,
            origin_lng,
            destination_lat,
            destination_lng,
        ]):
            return jsonify({
                "success": False,
                "message": (
                    "출발지 또는 목적지 "
                    "좌표가 없습니다."
                )
            }), 400

        data = call_kakao_route(
            origin_lat,
            origin_lng,
            destination_lat,
            destination_lng
        )

        return jsonify({
            "success": True,
            "data": data
        })

    except Exception as e:
        print(
            "KAKAO ROUTE ERROR:",
            e
        )

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route(
    "/api/hospital-etas",
    methods=["POST"]
)
def kakao_hospital_etas_api():
    """
    지도팀의 여러 병원 ETA 계산 API를
    Flask 백엔드로 통합한 API.
    """

    try:
        if not KAKAO_REST_API_KEY:
            return jsonify({
                "success": False,
                "message": (
                    ".env에 KAKAO_REST_API_KEY가 "
                    "설정되어 있지 않습니다."
                )
            }), 500

        body = (
            request.get_json()
            or {}
        )

        origin = (
            body.get("origin")
            or {}
        )

        hospitals = (
            body.get("hospitals")
            or []
        )

        origin_lat = to_number(
            origin.get("lat")
        )

        origin_lng = to_number(
            origin.get("lng")
        )

        if not is_valid_korean_coordinate(
            origin_lat,
            origin_lng
        ):
            return jsonify({
                "success": False,
                "message": (
                    "환자 위치 좌표가 올바르지 않습니다."
                )
            }), 400

        if (
            not isinstance(
                hospitals,
                list
            )
            or len(hospitals) == 0
            or len(hospitals) > 30
        ):
            return jsonify({
                "success": False,
                "message": (
                    "병원 후보는 1개 이상 "
                    "30개 이하로 보내주세요."
                )
            }), 400

        destinations = []

        for (
            index,
            hospital,
        ) in enumerate(
            hospitals
        ):
            lat = to_number(
                hospital.get(
                    "lat"
                )
            )

            lng = to_number(
                hospital.get(
                    "lng"
                )
            )

            if not is_valid_korean_coordinate(
                lat,
                lng
            ):
                return jsonify({
                    "success": False,
                    "message": (
                        f"{index + 1}번째 "
                        "병원 좌표가 올바르지 않습니다."
                    )
                }), 400

            destinations.append(
                {
                    "key": str(index),
                    "x": float(lng),
                    "y": float(lat),
                }
            )

        response = requests.post(
            "https://apis-navi.kakaomobility.com/v1/destinations/directions",
            headers={
                "Authorization": (
                    f"KakaoAK "
                    f"{KAKAO_REST_API_KEY}"
                ),
                "Content-Type": (
                    "application/json"
                ),
            },
            json={
                "origin": {
                    "x": float(
                        origin_lng
                    ),
                    "y": float(
                        origin_lat
                    ),
                },
                "destinations": (
                    destinations
                ),
                "radius": 10000,
                "priority": "TIME",
                "roadevent": 0,
            },
            timeout=(
                KAKAO_REQUEST_TIMEOUT_SEC
            ),
        )

        try:
            data = response.json()

        except Exception:
            data = {
                "raw": (
                    response.text[
                        :1000
                    ]
                )
            }

        if (
            response.status_code
            >= 400
        ):
            return jsonify({
                "success": False,
                "message": (
                    "카카오모빌리티 다중 목적지 "
                    "ETA 조회에 실패했습니다."
                ),
                "kakaoError": data,
            }), response.status_code

        return jsonify({
            "success": True,
            "data": data
        })

    except Exception as e:
        print(
            "KAKAO HOSPITAL ETAS ERROR:",
            e
        )

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route("/api/config")
def frontend_config_api():
    """
    프론트엔드에서 필요한 공개 설정값만 전달한다.

    JavaScript 키는 브라우저에서
    카카오맵 SDK를 로드하는 데 필요한 공개 키이다.
    REST API 키/Admin 키는 절대 내려보내지 않는다.
    """

    return jsonify({
        "success": True,
        "kakao_javascript_key": (
            KAKAO_JAVASCRIPT_KEY
            or ""
        ),
        "db_manager_url": DB_MANAGER_URL,
    })


# =========================================================
# 13. 지도팀 정적 파일 제공
# =========================================================

@app.route(
    "/js/<path:filename>"
)
def serve_js(filename):
    return send_from_directory(
        "js",
        filename
    )


@app.route(
    "/css/<path:filename>"
)
def serve_css(filename):
    return send_from_directory(
        "css",
        filename
    )


@app.route("/")
def index():
    # 실제 프로젝트 폴더의 HTML 파일명이
    # "server.ai.html"(점 포함)이므로 그에 맞춘다.
    return send_from_directory(
        ".",
        "server.ai.html"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)

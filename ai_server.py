import json
import os
from tempfile import NamedTemporaryFile

# ---------------------------------------------------------
# Render 등 클라우드 배포 환경용 Google Cloud STT 인증 설정
# ---------------------------------------------------------
google_json_str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if google_json_str:
    # Render 환경변수(GOOGLE_APPLICATION_CREDENTIALS_JSON)의 JSON 문자열을
    # 임시 .json 파일로 저장하여 Google SDK가 인식할 수 있도록 경로를 설정합니다.
    temp_key_file = NamedTemporaryFile(delete=False, suffix=".json")
    temp_key_file.write(google_json_str.encode("utf-8"))
    temp_key_file.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_key_file.name


from dotenv import load_dotenv #깃허브에선 지움

# .env를 Google/gRPC 라이브러리 import 전에 읽어 DNS resolver 설정이
# 실시간 STT 채널 생성에도 확실히 적용되도록 한다.
load_dotenv() #깃허브에선 지움
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

from google import genai
from google.genai import types
import google.auth
from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions

from pydantic import BaseModel
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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")

# 여러 개의 Gemini API 키를 순환 사용하기 위한 설정.
# .env에 GEMINI_API_KEYS="키1,키2,키3,키4" 처럼 콤마로 구분해서 넣으면 된다.
# GEMINI_API_KEYS가 없으면 기존 GEMINI_API_KEY 하나만 사용(하위 호환).
GEMINI_API_KEYS = [
    key.strip()
    for key in os.getenv("GEMINI_API_KEYS", "").split(",")
    if key.strip()
] or ([GEMINI_API_KEY] if GEMINI_API_KEY else [])

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
# 병원 추천/Gemini 환자 분석 로직에는 영향을 주지 않는다.
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

if not GEMINI_API_KEYS:
    raise ValueError("GEMINI_API_KEY(또는 GEMINI_API_KEYS)가 .env에 없습니다.")

# 키마다 하나씩 클라이언트를 미리 만들어두고, 쿼터 초과 시 다음 클라이언트로 순환한다.
_gemini_clients = [genai.Client(api_key=key) for key in GEMINI_API_KEYS]
_gemini_key_index = 0
_gemini_key_lock = Lock()

print(f"[GEMINI] {len(_gemini_clients)}개의 API 키를 순환 사용합니다.")

app = Flask(__name__, static_folder=".")
CORS(app)
sock = Sock(app)

# Streamlit에서 Excel 저장 없이 넘겨받는 최신 병원 DB.
# Flask가 재시작되면 사라지고, 그때는 아래 기존 Excel fallback 로직을 그대로 사용한다.
RUNTIME_HOSPITAL_DF = None
RUNTIME_HOSPITAL_META = {"source": None, "synced_at": None}
RUNTIME_DB_LOCK = Lock()

# 프론트에서 GPS 위치를 못 받아왔을 때 사용하는 기본 위치(대전광역시청).
# 실제 서비스에서는 반드시 구급차의 실시간 GPS 좌표를 프론트에서 전달해야 한다.
DEFAULT_AMBULANCE_LAT = 36.3504
DEFAULT_AMBULANCE_LNG = 127.3845

# 카카오모빌리티 REST API 키. 기존 지도팀의 server.js 역할을 Flask 백엔드 안으로 합친다.
# .env에 KAKAO_REST_API_KEY가 있어야 실제 도로 경로/실시간 교통 ETA를 조회할 수 있다.
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")
KAKAO_JAVASCRIPT_KEY = os.getenv("KAKAO_JAVASCRIPT_KEY")
KAKAO_REQUEST_TIMEOUT_SEC = 15

# 환자 위치 랜덤 생성 범위. 지도팀 코드의 전국 시연용 반경과 같은 의미다.
PATIENT_MIN_RADIUS_KM = 0.8
PATIENT_MAX_RADIUS_KM = 5.0

# 병원 사전 수용 신청/이송 시작 시 병원측에 전송한 환자 정보를 기록하는 로그 파일.
# 실제 병원 시스템 연동 API가 준비되기 전까지는 이 파일이 "전송 기록"의 역할을 한다.
DISPATCH_LOG_FILE = "dispatch_log.jsonl"

# 실제 거절/재이송 이력이 쌓이기 전까지 사용하는 보수적 프록시 가정치.
ASSUMED_REJECTION_RESEARCH_MIN = 20.0
ASSUMED_RETRANSFER_MIN = 30.0
MIN_EXPECTED_WAIT_MIN = 5.0
CONGESTION_WAIT_RANGE_MIN = 25.0

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
        "description": "서로 중복이 적은 ETA, 혼잡도, 전문치료 여유도, 데이터 신뢰도를 고정 가중치로 평가합니다.",
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
        "description": "C1과 동일한 기준을 사용하되 KTAS/골든타임에 따라 가중치만 변경합니다.",
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

MODEL_INFO = {
    "A": {
        "name": "Model A",
        "label": "가장 가까운 병원 추천",
        "description": "수용 가능 판정을 통과한 병원 중 구급차 현재 위치에서 직선거리가 가장 가까운 병원 순으로 정렬합니다.",
    },
    "B": {
        "name": "Model B",
        "label": "MILP 혼합정수선형계획",
        "description": "예상 치료시작 지연, 임상 적합성, 정보 신뢰도와 골든타임 위반을 반영합니다. 다중 환자 API에서는 병상 용량까지 동시에 최적화합니다.",
    },
    "C": {
        "name": "Model C",
        "label": "TOPSIS 다기준 의사결정",
        "description": "동일한 후보 병원과 공통 TOPSIS 엔진으로 C0/C1/C2/C3 변형을 비교합니다. 기본값은 독립적 기준을 사용하는 C1입니다.",
    },
    "D": {
        "name": "Model D",
        "label": "가중치 종합 점수",
        "description": "수용점수, 이동거리, 전문의 수, 병원 혼잡도를 0~1로 정규화한 뒤 사전에 정의한 가중치로 합산한 종합 점수 순으로 정렬합니다.",
    },
}


# =========================================================
# 2. 환자 정보 구조
# =========================================================

class PatientInfo(BaseModel):
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
            }
        )

    except Exception as e:
        print("RUNTIME DB SYNC ERROR:", e)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


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
    Gemini 분석/병원 추천/지도 로직은 호출하지 않는다.
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
# 4. Gemini 환자 상태 분석
# =========================================================

def _is_gemini_503(error):
    """Gemini의 일시적 서버 과부하/UNAVAILABLE(503)인지 보수적으로 판별한다."""
    for attr in ("code", "status_code", "status"):
        value = getattr(error, attr, None)
        if value == 503 or str(value).strip() == "503":
            return True

    message = str(error).upper()
    return (
        "503" in message
        and ("UNAVAILABLE" in message or "HIGH DEMAND" in message)
    )


def _is_gemini_quota_exceeded(error):
    """무료 API 키의 사용량(쿼터)을 다 써서 나는 429/RESOURCE_EXHAUSTED 오류인지 판별한다."""
    for attr in ("code", "status_code", "status"):
        value = getattr(error, attr, None)
        if value == 429 or str(value).strip() == "429":
            return True

    message = str(error).upper()
    return (
        "429" in message
        or "RESOURCE_EXHAUSTED" in message
        or "QUOTA" in message
        or "RATE LIMIT" in message
    )


def _current_gemini_key_index():
    with _gemini_key_lock:
        return _gemini_key_index


def _advance_gemini_key(from_index):
    """쿼터가 소진된 키를 다음 키로 넘긴다. 다른 요청이 이미 넘겨놨다면 중복으로 넘기지 않는다."""
    global _gemini_key_index
    with _gemini_key_lock:
        if _gemini_key_index == from_index:
            _gemini_key_index = (_gemini_key_index + 1) % len(_gemini_clients)
        return _gemini_key_index


def _generate_patient_with_model(model_name, prompt):
    """
    준비된 Gemini API 키를 순환하며 환자 분석을 요청한다.
    현재 키가 쿼터 초과(429)면 다음 키로 넘어가서 같은 모델로 재시도하고,
    모든 키를 다 써봤는데도 실패하면 마지막 오류를 그대로 올린다.
    """
    last_error = None

    for _ in range(len(_gemini_clients)):
        key_index = _current_gemini_key_index()
        active_client = _gemini_clients[key_index]

        try:
            return active_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PatientInfo,
                ),
            )
        except Exception as error:
            last_error = error
            if not _is_gemini_quota_exceeded(error):
                raise

            print(
                f"GEMINI QUOTA EXCEEDED: key #{key_index + 1}/{len(_gemini_clients)} "
                f"소진 -> 다음 키로 전환합니다."
            )
            _advance_gemini_key(key_index)

    # 준비된 키를 모두 순환해도 전부 쿼터 초과였던 경우
    raise last_error


def analyze_patient(text: str):
    prompt = f"""
너는 응급의료 환자 상태를 구조화하는 AI이다.
아래 구급대원이 입력한 자연어 환자 상태를 분석해서 JSON으로 변환해라.

반드시 아래 필드를 채워라.

suspected_category:
- 환자 상태 카테고리
- 예: 뇌졸중/뇌출혈 의심, 심근경색 의심, 다발성 외상, 고위험 산모, 소아 중증 탈수, 호흡곤란/중증감염, 기타

ktas:
- KTAS 1~5 정수

golden_time_min:
- 골든타임 분 단위 정수

specialists:
- 필요 전문의/진료과 리스트
- 예: 응급의학과, 신경과, 신경외과, 심장내과, 흉부외과, 외과, 정형외과, 산부인과, 소아청소년과

equipment:
- 필요 장비 리스트
- 예: CT, MRI, CAG, 혈관조영, 인공호흡기

req_icu:
- ICU 필요 여부 boolean

req_or:
- 수술실 필요 여부 boolean

req_er_bed:
- 응급실 병상 필요 여부 boolean

req_severe_acceptance:
- 중증환자 수용 가능성 확인 필요 여부 boolean

중요:
- 병원 추천은 하지 마라.
- 환자 상태에서 필요한 조건만 구조화해라.
- 잘 모르겠으면 응급 상황에서 안전한 쪽으로 판단해라.

환자 상태:
{text}
"""

    try:
        response = _generate_patient_with_model(
            GEMINI_MODEL,
            prompt,
        )
        used_model = GEMINI_MODEL

    except Exception as primary_error:
        if (
            not _is_gemini_503(primary_error)
            or not GEMINI_FALLBACK_MODEL
            or GEMINI_FALLBACK_MODEL == GEMINI_MODEL
        ):
            raise

        print(
            f"GEMINI 503: {GEMINI_MODEL} unavailable -> "
            f"fallback to {GEMINI_FALLBACK_MODEL}"
        )

        response = _generate_patient_with_model(
            GEMINI_FALLBACK_MODEL,
            prompt,
        )
        used_model = GEMINI_FALLBACK_MODEL

    patient = PatientInfo.model_validate_json(response.text)
    result = patient.model_dump()

    print(f"GEMINI ANALYSIS MODEL: {used_model}")

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


def get_patient_priority_profile(patient):
    """환자 중증도와 골든타임에 따라 모델의 선호도 프로파일을 반환합니다."""
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


def compute_data_freshness_score(hospital, now=None):
    """실시간 데이터 최신성 점수를 0~1로 환산합니다."""
    timestamp = None
    for col in ["data_fetched_at", "hvidate", "master_last_updated_at", "departments_last_checked_at"]:
        value = hospital.get(col)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        text = str(value).strip()
        if not text or text.lower() in ["none", "nan"]:
            continue

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
    if age_min <= 15: return 1.0
    if age_min <= 60: return 0.85
    if age_min <= 180: return 0.65
    if age_min <= 720: return 0.4
    if age_min <= 1440: return 0.25
    return 0.1


def compute_operational_estimates(hospital, eta_min, congestion_score, now=None):
    """대기, 거절, 재이송 지연 및 수용확률 추정치를 계산합니다."""
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
    """필수 전문과 충족 후 여유 전문의 수를 계산합니다."""
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


def attach_routing_fields(hospital_df, ambulance_lat, ambulance_lng, patient=None, as_of=None):
    """이동거리/ETA/혼잡도 및 예상 치료시작 지연 정보 필드를 결합합니다."""
    if hospital_df.empty:
        hospital_df["distance_km"] = []
        hospital_df["eta_min"] = []
        hospital_df["congestion_score"] = []
        hospital_df["data_freshness_score"] = []
        hospital_df["data_reliability_score"] = []
        hospital_df["estimated_acceptance_probability"] = []
        hospital_df["expected_wait_min"] = []
        hospital_df["expected_rejection_delay_min"] = []
        hospital_df["expected_retransfer_delay_min"] = []
        hospital_df["effective_treatment_delay_min"] = []
        hospital_df["specialty_margin"] = []
        hospital_df["specialty_margin_basis"] = []
        return hospital_df

    distances, etas, congestions = [], [], []
    operational_estimates, specialty_margins, specialty_margin_bases = [], [], []

    for _, row in hospital_df.iterrows():
        distance_km, eta_min = compute_distance_and_eta(row, ambulance_lat, ambulance_lng)
        congestion_score = compute_congestion_score(row)
        distances.append(distance_km)
        etas.append(eta_min)
        congestions.append(congestion_score)
        operational_estimates.append(compute_operational_estimates(row, eta_min, congestion_score, now=as_of))
        specialty_margin, specialty_margin_basis = compute_specialty_margin(row, patient or {})
        specialty_margins.append(specialty_margin)
        specialty_margin_bases.append(specialty_margin_basis)

    hospital_df = hospital_df.copy()
    hospital_df["distance_km"] = distances
    hospital_df["eta_min"] = etas
    hospital_df["congestion_score"] = congestions
    for col in operational_estimates[0]:
        hospital_df[col] = [estimate[col] for estimate in operational_estimates]
    hospital_df["specialty_margin"] = specialty_margins
    hospital_df["specialty_margin_basis"] = specialty_margin_bases

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


def rank_model_a(
    acceptable_df
):
    """
    Model A:
    가장 가까운 병원 추천
    """

    if acceptable_df.empty:
        return acceptable_df

    df = acceptable_df.copy()

    df["_sort_distance"] = (
        df["distance_km"].apply(
            lambda v: (
                v
                if v is not None
                else float("inf")
            )
        )
    )

    df["_sort_score"] = (
        pd.to_numeric(
            df["acceptance_score"],
            errors="coerce"
        ).fillna(0)
    )

    df = df.sort_values(
        by=[
            "_sort_distance",
            "_sort_score",
        ],
        ascending=[
            True,
            False,
        ],
        kind="mergesort",
    ).drop(
        columns=[
            "_sort_distance",
            "_sort_score",
        ]
    )

    df[
        "model_rank_explanation"
    ] = df[
        "distance_km"
    ].apply(
        lambda v: (
            f"구급차 위치로부터 약 {v}km"
            if v is not None
            else (
                "거리 정보 없음"
                "(병원 좌표 미확보)"
            )
        )
    )

    return df.reset_index(
        drop=True
    )


def rank_model_b(acceptable_df, patient=None, weights=None):
    """Model B: 치료 지연, 데이터 신뢰도 및 골든타임 패널티를 반영한 MILP 순위 산정."""
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

    remaining_idx = list(df.index)
    ranked_idx = []

    if PULP_AVAILABLE:
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


def get_topsis_criteria(variant=DEFAULT_TOPSIS_VARIANT, patient=None):
    """C0~C3 변형 TOPSIS 기준을 추출합니다."""
    variant = str(variant or DEFAULT_TOPSIS_VARIANT).strip().upper()
    if variant not in TOPSIS_VARIANTS:
        raise ValueError(f"알 수 없는 TOPSIS 변형입니다: {variant}")

    config = TOPSIS_VARIANTS[variant]
    base = TOPSIS_VARIANTS[config["criteria_from"]]["criteria"] if config.get("criteria_from") else config["criteria"]

    criteria = {name: {"benefit": bool(v["benefit"]), "weight": float(v["weight"])} for name, v in base.items()}

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
    """TOPSIS 변형 메타데이터 구조 생성."""
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
            {"name": k, "direction": "benefit" if v["benefit"] else "cost", "weight": v["weight"]}
            for k, v in criteria.items()
        ],
    }


def _topsis_closeness(df, criteria):
    """공통 TOPSIS 순수 계산 엔진."""
    if df.empty:
        return pd.Series(dtype=float, index=df.index)
    
    missing_columns = [name for name in criteria if name not in df.columns]
    if missing_columns:
        raise KeyError(f"TOPSIS 기준 컬럼이 없습니다: {', '.join(missing_columns)}")

    matrix = pd.DataFrame(index=df.index)
    for column, config in criteria.items():
        values = pd.to_numeric(df[column], errors="coerce")
        fill_value = (values.min() if config["benefit"] else values.max()) if values.notna().any() else 0.0
        matrix[column] = values.fillna(fill_value)

    vector_norm = np.sqrt((matrix ** 2).sum()).replace(0, np.nan)
    normalized = matrix.divide(vector_norm, axis=1).fillna(0.0)
    weighted = normalized.copy()
    for column, config in criteria.items():
        weighted[column] = normalized[column] * float(config["weight"])

    ideal_best = {k: weighted[k].max() if v["benefit"] else weighted[k].min() for k, v in criteria.items()}
    ideal_worst = {k: weighted[k].min() if v["benefit"] else weighted[k].max() for k, v in criteria.items()}

    distance_best = np.sqrt(sum((weighted[k] - ideal_best[k]) ** 2 for k in criteria))
    distance_worst = np.sqrt(sum((weighted[k] - ideal_worst[k]) ** 2 for k in criteria))
    denominator = (distance_best + distance_worst).replace(0, np.nan)
    return (distance_worst / denominator).fillna(0.5)


def rank_model_c(acceptable_df, patient=None, variant=DEFAULT_TOPSIS_VARIANT):
    """C0/C1/C2/C3 변형 TOPSIS 적용 순위."""
    if acceptable_df.empty:
        return acceptable_df

    patient = patient or {}
    variant = str(variant or DEFAULT_TOPSIS_VARIANT).strip().upper()
    criteria, profile = get_topsis_criteria(variant, patient)
    metadata = get_topsis_variant_metadata(variant, patient)

    df = acceptable_df.copy()
    df["_topsis_closeness"] = _topsis_closeness(df, criteria)
    df["_hospital_name_tiebreaker"] = df.apply(get_hospital_name, axis=1)
    df = df.sort_values(by=["_topsis_closeness", "_hospital_name_tiebreaker"], ascending=[False, True], kind="mergesort")

    profile_text = f", {profile['label']} 가중치" if profile else ""
    experimental_text = ", 실험안" if metadata["experimental"] else ""
    df["model_rank_explanation"] = df["_topsis_closeness"].apply(
        lambda v: f"TOPSIS {variant} 근접도 {round(float(v), 3)} ({metadata['label']}{profile_text}{experimental_text})"
    )
    df["topsis_closeness"] = df["_topsis_closeness"].round(6)
    df["topsis_variant"] = variant
    df["topsis_variant_label"] = metadata["label"]
    df["topsis_experimental"] = metadata["experimental"]
    df = df.drop(columns=["_topsis_closeness", "_hospital_name_tiebreaker"])
    return df.reset_index(drop=True)


def apply_ranking_model(model_key, acceptable_df, patient=None, topsis_variant=DEFAULT_TOPSIS_VARIANT):
    model_key = (model_key or "A").strip().upper()
    if model_key == "B": return rank_model_b(acceptable_df, patient=patient), "B"
    if model_key == "C": return rank_model_c(acceptable_df, patient=patient, variant=topsis_variant), "C"
    if model_key == "D": return rank_model_d(acceptable_df), "D"
    return rank_model_a(acceptable_df), "A"


# 다중 환자 배치용 함수 구현체
def _batch_hospital_key(hospital, fallback_index=None):
    hpid = str(hospital.get("hpid") or "").strip()
    if hpid and hpid.lower() != "nan": return f"hpid:{hpid}"
    name = get_hospital_name(hospital)
    lat, lng = get_hospital_lat_lng(hospital)
    if lat is not None and lng is not None:
        return f"name:{name}|{round(float(lat), 5)}|{round(float(lng), 5)}"
    return f"name:{name}|row:{fallback_index if fallback_index is not None else 'unknown'}"


def _batch_capacity(hospital):
    hvec = to_number(hospital.get("hvec"))
    return max(1, int(math.floor(hvec))) if hvec is not None else 1


def _assignment_cost(candidate, patient):
    profile = get_patient_priority_profile(patient)
    delay = to_number(candidate.get("effective_treatment_delay_min"))
    delay = 999.0 if delay is None else delay
    clinical = min(1.0, max(0.0, (to_number(candidate.get("acceptance_score")) or 0.0) / 100.0))
    reliability = min(1.0, max(0.0, to_number(candidate.get("data_reliability_score")) or 0.0))
    return profile["severity_weight"] * (delay + 35.0 * (1.0 - clinical) + 12.0 * (1.0 - reliability))


def solve_batch_candidate_assignment(candidate_rows_by_patient, patients_by_id):
    patient_ids = list(patients_by_id)
    capacities, candidate_lookup = {}, {}

    for patient_id in patient_ids:
        for candidate in candidate_rows_by_patient.get(patient_id, []):
            hospital_id = candidate["_batch_hospital_id"]
            candidate_lookup[(patient_id, hospital_id)] = candidate
            capacities[hospital_id] = max(capacities.get(hospital_id, 0), int(candidate.get("_batch_capacity", 1)))

    assignments = []
    solver_name = "pulp_cbc" if PULP_AVAILABLE else "severity_greedy_fallback"
    objective_value = None

    if PULP_AVAILABLE:
        problem = pulp.LpProblem("multi_patient_hospital_assignment", pulp.LpMinimize)
        x = {key: pulp.LpVariable(f"x_{p_idx}_{h_idx}", cat="Binary") for p_idx, key in enumerate(candidate_lookup) for h_idx in [0]}
        unassigned = {patient_id: pulp.LpVariable(f"unassigned_{idx}", cat="Binary") for idx, patient_id in enumerate(patient_ids)}
        late = {patient_id: pulp.LpVariable(f"late_{idx}", lowBound=0) for idx, patient_id in enumerate(patient_ids)}

        for patient_id in patient_ids:
            patient_keys = [key for key in candidate_lookup if key[0] == patient_id]
            problem += pulp.lpSum(x[key] for key in patient_keys) + unassigned[patient_id] == 1
            golden_time = to_number(patients_by_id[patient_id].get("golden_time_min"))
            if golden_time and patient_keys:
                problem += late[patient_id] >= pulp.lpSum(
                    ((to_number(candidate_lookup[key].get("effective_treatment_delay_min")) or 999.0) - golden_time) * x[key]
                    for key in patient_keys
                )
            else:
                problem += late[patient_id] == 0

        for hospital_id, capacity in capacities.items():
            hospital_keys = [key for key in candidate_lookup if key[1] == hospital_id]
            problem += pulp.lpSum(x[key] for key in hospital_keys) <= capacity

        assignment_cost = pulp.lpSum(_assignment_cost(candidate_lookup[k], patients_by_id[k[0]]) * x[k] for k in candidate_lookup)
        late_cost = pulp.lpSum(6.0 * get_patient_priority_profile(patients_by_id[p_id])["severity_weight"] * late[p_id] for p_id in patient_ids)
        unassigned_cost = pulp.lpSum(10000.0 * get_patient_priority_profile(patients_by_id[p_id])["severity_weight"] * unassigned[p_id] for p_id in patient_ids)
        problem += assignment_cost + late_cost + unassigned_cost
        problem.solve(pulp.PULP_CBC_CMD(msg=False))
        objective_value = pulp.value(problem.objective)

        for patient_id in patient_ids:
            chosen_key = next((k for k in candidate_lookup if k[0] == patient_id and (pulp.value(x[k]) or 0) > 0.5), None)
            assignments.append((patient_id, candidate_lookup.get(chosen_key) if chosen_key else None))
    else:
        remaining_capacity = dict(capacities)
        ordered_patients = sorted(patient_ids, key=lambda p_id: get_patient_priority_profile(patients_by_id[p_id])["severity_weight"], reverse=True)
        greedy_result = {}
        for patient_id in ordered_patients:
            available = [c for c in candidate_rows_by_patient.get(patient_id, []) if remaining_capacity.get(c["_batch_hospital_id"], 0) > 0]
            chosen = min(available, key=lambda row: _assignment_cost(row, patients_by_id[patient_id])) if available else None
            greedy_result[patient_id] = chosen
            if chosen: remaining_capacity[chosen["_batch_hospital_id"]] -= 1
        assignments = [(p_id, greedy_result.get(p_id)) for p_id in patient_ids]

    output = []
    for patient_id, candidate in assignments:
        patient = patients_by_id[patient_id]
        if candidate is None:
            output.append({"patient_id": patient_id, "assigned": False, "reason": "임상 적합 후보 또는 가용 병상이 없어 수동 조정이 필요합니다."})
            continue

        effective_delay = to_number(candidate.get("effective_treatment_delay_min"))
        golden_time = to_number(patient.get("golden_time_min"))
        output.append({
            "patient_id": patient_id,
            "assigned": True,
            "hospital_id": candidate["_batch_hospital_id"],
            "hospital_name": get_hospital_name(candidate),
            "hospital_capacity": candidate.get("_batch_capacity"),
            "eta_min": candidate.get("eta_min"),
            "effective_treatment_delay_min": effective_delay,
            "golden_time_min": golden_time,
            "golden_time_violation_min": round(max(0.0, effective_delay - golden_time), 1) if effective_delay is not None and golden_time else None,
            "acceptance_score": candidate.get("acceptance_score"),
            "estimated_acceptance_probability": candidate.get("estimated_acceptance_probability"),
            "data_reliability_score": candidate.get("data_reliability_score"),
            "assignment_cost": round(_assignment_cost(candidate, patient), 3),
        })

    return {
        "solver": solver_name,
        "objective_value": round(float(objective_value), 3) if objective_value is not None else None,
        "patient_count": len(patient_ids),
        "assigned_count": sum(1 for item in output if item["assigned"]),
        "unassigned_count": sum(1 for item in output if not item["assigned"]),
        "assignments": output,
    }


def optimize_batch_assignments(patient_requests, hospital_df):
    """다중 환자 병상 배치 최적화 진입점 함수."""
    hospitals = hospital_df.copy().reset_index(drop=True)
    hospitals["_batch_hospital_id"] = [_batch_hospital_key(row, idx) for idx, row in hospitals.iterrows()]

    patients_by_id, candidate_rows_by_patient = {}, {}
    for index, item in enumerate(patient_requests):
        patient_id = str(item.get("patient_id") or f"patient-{index + 1}")
        if patient_id in patients_by_id:
            raise ValueError(f"중복된 patient_id입니다: {patient_id}")
        patient = enrich_patient_by_category(item["patient"])
        patients_by_id[patient_id] = patient
        amb_lat = to_number(item.get("ambulance_lat")) or DEFAULT_AMBULANCE_LAT
        amb_lng = to_number(item.get("ambulance_lng")) or DEFAULT_AMBULANCE_LNG

        acceptable_df, _, _ = filter_acceptable_hospitals(hospitals, patient)
        acceptable_df = attach_routing_fields(acceptable_df, amb_lat, amb_lng, patient=patient)
        if acceptable_df.empty:
            candidate_rows_by_patient[patient_id] = []
            continue

        acceptable_df["_batch_capacity"] = acceptable_df.apply(_batch_capacity, axis=1)
        candidate_rows_by_patient[patient_id] = acceptable_df.to_dict(orient="records")

    return solve_batch_candidate_assignment(candidate_rows_by_patient, patients_by_id)

def rank_model_d(
    acceptable_df,
    weights=None
):
    """
    Model D:
    가중치 종합 점수.
    """

    if acceptable_df.empty:
        return acceptable_df

    weights = weights or {
        "score": 0.45,
        "distance": 0.25,
        "specialist": 0.15,
        "congestion": 0.15,
    }

    df = acceptable_df.copy()

    norm_score = (
        _normalize_series(
            df["acceptance_score"],
            higher_is_better=True
        )
    )

    norm_distance = (
        _normalize_series(
            df["distance_km"],
            higher_is_better=False
        )
    )

    norm_specialist = (
        _normalize_series(
            df[
                "required_specialist_count_total"
            ],
            higher_is_better=True
        )
    )

    norm_congestion = (
        _normalize_series(
            df["congestion_score"],
            higher_is_better=False
        )
    )

    df[
        "_weighted_total"
    ] = (
        weights["score"]
        * norm_score
        + weights["distance"]
        * norm_distance
        + weights["specialist"]
        * norm_specialist
        + weights["congestion"]
        * norm_congestion
    )

    df["_sort_score"] = (
        pd.to_numeric(
            df["acceptance_score"],
            errors="coerce"
        ).fillna(0)
    )

    df = df.sort_values(
        by=[
            "_weighted_total",
            "_sort_score",
        ],
        ascending=[
            False,
            False,
        ],
        kind="mergesort",
    )

    df[
        "model_rank_explanation"
    ] = df[
        "_weighted_total"
    ].apply(
        lambda v: (
            "가중합 종합점수 "
            f"{round(float(v), 3)} "
            "(수용점수 45% · 거리 25% · "
            "전문의수 15% · 혼잡도 15%)"
        )
    )

    df = df.drop(
        columns=[
            "_weighted_total",
            "_sort_score",
        ]
    )

    return df.reset_index(
        drop=True
    )


# =========================================================
# 5-3. 시스템 성능 검증 지표
# =========================================================

def compute_performance_metrics(result_df, acceptable_df, top_df, patient, elapsed_ms):
    total = len(result_df)
    acceptable_count = len(acceptable_df)

    acceptance_success_rate = round((acceptable_count / total) * 100, 1) if total else 0.0

    golden_time = to_number(patient.get("golden_time_min"))
    eta_series = pd.to_numeric(top_df["eta_min"], errors="coerce").dropna() if "eta_min" in top_df else pd.Series(dtype=float)
    treatment_delay_series = (
        pd.to_numeric(top_df["effective_treatment_delay_min"], errors="coerce").dropna()
        if "effective_treatment_delay_min" in top_df
        else pd.Series(dtype=float)
    )

    if golden_time and not treatment_delay_series.empty:
        within_golden = (treatment_delay_series <= golden_time).sum()
        golden_time_compliance_rate = round((within_golden / len(treatment_delay_series)) * 100, 1)
    else:
        golden_time_compliance_rate = None

    avg_eta = round(float(eta_series.mean()), 1) if not eta_series.empty else None
    p95_eta = round(float(np.percentile(eta_series, 95)), 1) if len(eta_series) > 0 else None
    avg_treatment_delay = round(float(treatment_delay_series.mean()), 1) if not treatment_delay_series.empty else None

    top1_acceptance_probability = None
    top1_treatment_delay = None
    if not top_df.empty:
        top1_acceptance_probability = to_number(top_df.iloc[0].get("estimated_acceptance_probability"))
        top1_treatment_delay = to_number(top_df.iloc[0].get("effective_treatment_delay_min"))

    congestion_series = pd.to_numeric(top_df["congestion_score"], errors="coerce").dropna() if "congestion_score" in top_df else pd.Series(dtype=float)
    avg_congestion_pct = round(float(congestion_series.mean()) * 100, 1) if not congestion_series.empty else None

    return {
        "acceptance_success_rate_pct": acceptance_success_rate,
        "golden_time_compliance_rate_pct": golden_time_compliance_rate,
        "avg_eta_min": avg_eta,
        "p95_eta_min": p95_eta,
        "avg_effective_treatment_delay_min": avg_treatment_delay,
        "top1_effective_treatment_delay_min": round(top1_treatment_delay, 1) if top1_treatment_delay is not None else None,
        "top1_estimated_acceptance_probability_pct": (
            round(top1_acceptance_probability * 100, 1) if top1_acceptance_probability is not None else None
        ),
        "hospital_congestion_pct": avg_congestion_pct,
        "computation_time_ms": round(elapsed_ms, 1),
        "excluded_metrics": [
            {
                "name": "재매칭률",
                "reason": "병원의 실제 거절/재요청 이력을 받는 연동이 아직 없어 항상 0으로만 계산되어 의미가 없습니다.",
            },
            {
                "name": "도착 시 수용률",
                "reason": "구급차 도착 시점의 병원 최종 확정 응답을 받는 연동이 아직 없어 계산할 수 없습니다.",
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
            "탈수",
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
            "탈수",
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

@app.route("/api/recommend-hospitals", methods=["POST"])
def recommend_hospitals_api():
    start_time = time.perf_counter()

    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"success": False, "error": "text가 필요합니다."}), 400

        raw_patient = analyze_patient(data["text"])
        patient = enrich_patient_by_category(raw_patient)

        ambulance_lat = to_number(data.get("ambulance_lat")) or DEFAULT_AMBULANCE_LAT
        ambulance_lng = to_number(data.get("ambulance_lng")) or DEFAULT_AMBULANCE_LNG
        model_key = data.get("model", "A")
        topsis_variant = str(data.get("topsis_variant") or DEFAULT_TOPSIS_VARIANT).strip().upper()

        hospital_df, excel_file, sheet_name = load_hospital_db()
        acceptable_df, rejected_df, result_df = filter_acceptable_hospitals(hospital_df, patient)

        # 환자 정보(patient)를 인자로 추가 전달하여 운영 지연 및 가중치를 반영합니다.
        acceptable_df = attach_routing_fields(acceptable_df, ambulance_lat, ambulance_lng, patient=patient)
        rejected_df = attach_routing_fields(rejected_df, ambulance_lat, ambulance_lng, patient=patient)

        # TOPSIS 변형 및 환자 중증도 정보를 반영합니다.
        ranked_acceptable_df, applied_model_key = apply_ranking_model(
            model_key,
            acceptable_df,
            patient=patient,
            topsis_variant=topsis_variant,
        )

        top_acceptable_df = ranked_acceptable_df.head(10).copy()
        top_rejected_df = rejected_df.head(max(0, 10 - len(top_acceptable_df))).copy()
        top_result_df = pd.concat([top_acceptable_df, top_rejected_df], ignore_index=True) if not top_rejected_df.empty else top_acceptable_df

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        performance_metrics = compute_performance_metrics(
            result_df=result_df,
            acceptable_df=acceptable_df,
            top_df=top_acceptable_df,
            patient=patient,
            elapsed_ms=elapsed_ms,
        )

        return jsonify(
            {
                "success": True,
                "source_file": excel_file,
                "sheet_name": sheet_name,
                "total_hospital_count": len(result_df),
                "display_hospital_count": len(top_result_df),
                "patient": patient,
                "acceptable_count": len(acceptable_df),
                "rejected_count": len(rejected_df),
                "display_acceptable_count": len(top_acceptable_df),
                "display_rejected_count": len(top_rejected_df),
                "acceptable_hospitals": dataframe_to_records(top_acceptable_df),
                "rejected_hospitals": dataframe_to_records(top_rejected_df),
                "all_results": dataframe_to_records(top_result_df),
                "applied_model": {**MODEL_INFO[applied_model_key], "key": applied_model_key},
                "topsis_variant": (
                    get_topsis_variant_metadata(topsis_variant, patient)
                    if applied_model_key == "C"
                    else None
                ),
                "patient_priority_profile": get_patient_priority_profile(patient),
                "ambulance_location": {"lat": ambulance_lat, "lng": ambulance_lng},
                "performance_metrics": performance_metrics,
            }
        )
    except Exception as e:
        print("RECOMMEND ERROR:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# 신규 추가: 다중 환자 병상 배치 최적화 API
@app.route("/api/optimize-batch", methods=["POST"])
def optimize_batch_api():
    start_time = time.perf_counter()
    try:
        body = request.get_json() or {}
        patient_items = body.get("patients") or []
        if not isinstance(patient_items, list) or not patient_items:
            return jsonify({"success": False, "error": "patients 배열이 필요합니다."}), 400
        if len(patient_items) > 30:
            return jsonify({"success": False, "error": "한 번에 최대 30명까지 배정할 수 있습니다."}), 400

        prepared = []
        for index, item in enumerate(patient_items):
            if item.get("patient"):
                structured = PatientInfo(**item["patient"])
                patient = structured.model_dump() if hasattr(structured, "model_dump") else structured.dict()
            elif str(item.get("text") or "").strip():
                patient = analyze_patient(item["text"])
            else:
                return jsonify({"success": False, "error": f"{index + 1}번째 환자에 text 또는 patient 정보가 필요합니다."}), 400

            prepared.append({
                "patient_id": str(item.get("patient_id") or f"patient-{index + 1}"),
                "patient": patient,
                "ambulance_lat": item.get("ambulance_lat"),
                "ambulance_lng": item.get("ambulance_lng"),
            })

        hospital_df, excel_file, sheet_name = load_hospital_db()
        result = optimize_batch_assignments(prepared, hospital_df)
        result.update({
            "success": True,
            "source_file": excel_file,
            "sheet_name": sheet_name,
            "applied_model": {**MODEL_INFO["B"], "key": "B"},
            "computation_time_ms": round((time.perf_counter() - start_time) * 1000, 1),
        })
        return jsonify(result)

    except Exception as e:
        print("BATCH OPTIMIZE ERROR:", e)
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

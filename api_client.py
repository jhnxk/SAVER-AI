import os
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
from dotenv import load_dotenv
from urllib.parse import urlencode

load_dotenv()

BASE_URL = "https://apis.data.go.kr/B552657/ErmctInfoInqireService"
DEFAULT_STAGE1 = "대전광역시"

TARGET_HOSPITALS = [
    "건양대학교병원",
    "대전병원",
    "대전보훈병원",
    "대전선병원",
    "대전성모병원",
    "대전을지대학교병원",
    "대전한국병원",
    "대청병원",
    "유성선병원",
    "충남대학교병원",
]


def _text(elem, tag):
    found = elem.find(tag)
    if found is None or found.text is None:
        return None
    return found.text.strip()


def _parse_xml_items(xml_text):
    root = ET.fromstring(xml_text)

    header = root.find(".//header")
    result_code = _text(header, "resultCode") if header is not None else None
    result_msg = _text(header, "resultMsg") if header is not None else None

    items = []
    for item in root.findall(".//item"):
        row = {
            child.tag: child.text.strip() if child.text else None
            for child in list(item)
        }
        items.append(row)

    return result_code, result_msg, pd.DataFrame(items)


def call_api(endpoint, params=None):
    """
    공공데이터포털 인증키가 이미 인코딩된 형태일 수 있으므로
    serviceKey는 params에 넣지 않고 URL에 직접 붙인다.
    나머지 파라미터만 urlencode로 인코딩한다.
    """
    service_key = os.getenv("SERVICE_KEY")
    if not service_key:
        raise RuntimeError(".env 파일에 SERVICE_KEY가 없습니다.")

    url = BASE_URL + endpoint

    normal_params = {
        "pageNo": "1",
        "numOfRows": "1000",
        **(params or {}),
    }

    query = urlencode(normal_params)
    full_url = f"{url}?serviceKey={service_key}&{query}"

    response = requests.get(full_url, timeout=20)
    response.raise_for_status()

    return _parse_xml_items(response.text)


def clean_yn_columns(df):
    """
    Y/N 컬럼에서 N1처럼 비표준값이 내려오는 경우를 정리한다.
    예:
    Y  -> Y
    N  -> N
    N1 -> N
    """
    yn_cols = [
        "hvctayn",
        "hvmriayn",
        "hvangioayn",
        "hvventiayn",
    ]

    df = df.copy()

    for col in yn_cols:
        if col in df.columns:
            def normalize(value):
                if pd.isna(value):
                    return None

                value = str(value).strip().upper()

                if value.startswith("Y"):
                    return "Y"
                if value.startswith("N"):
                    return "N"

                return value

            df[col] = df[col].apply(normalize)

    return df


def fetch_realtime_beds(stage1=DEFAULT_STAGE1):
    """
    응급실 실시간 가용병상정보 조회
    endpoint: /getEmrrmRltmUsefulSckbdInfoInqire
    """
    code, msg, df = call_api(
        "/getEmrrmRltmUsefulSckbdInfoInqire",
        {"STAGE1": stage1},
    )

    df = clean_yn_columns(df)

    df["api_result_code"] = code
    df["api_result_msg"] = msg
    df["data_fetched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return df


def fetch_hospital_list(stage1=DEFAULT_STAGE1):
    """
    응급의료기관 목록정보 조회
    endpoint: /getEgytListInfoInqire

    현재 실시간 업데이트 단계에서는 필수로 사용하지 않는다.
    """
    code, msg, df = call_api(
        "/getEgytListInfoInqire",
        {"Q0": stage1},
    )

    df["api_result_code"] = code
    df["api_result_msg"] = msg
    df["data_fetched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return df


def normalize_name(name):
    if pd.isna(name):
        return ""

    s = str(name)

    remove_tokens = [
        "의료법인",
        "학교법인",
        "재단법인",
        "사회복지법인",
        "영훈의료재단",
        "가톨릭대학교",
        "학교법인가톨릭학원",
    ]

    for token in remove_tokens:
        s = s.replace(token, "")

    s = s.replace(" ", "")
    return s


def match_target_hospitals(api_df):
    """
    API 결과에서 지정한 10개 병원만 추출한다.
    병원명 컬럼명이 API 응답마다 조금씩 다를 수 있어 후보 컬럼을 여러 개 확인한다.
    """
    df = api_df.copy()

    name_candidates = [
        "dutyName",
        "dutyname",
        "dutyNm",
        "dutyNm",
    ]

    name_col = next((c for c in name_candidates if c in df.columns), None)

    if name_col is None:
        out = pd.DataFrame({
            "target_hospital": TARGET_HOSPITALS,
            "match_status": "병원명컬럼없음",
        })
        out.attrs["reason"] = f"실제 컬럼: {list(df.columns)}"
        return out

    df["_norm_name"] = df[name_col].map(normalize_name)

    matched_rows = []

    for target in TARGET_HOSPITALS:
        target_norm = normalize_name(target)

        candidates = df[df["_norm_name"].str.contains(target_norm, na=False)].copy()

        # 대전선병원 / 유성선병원 혼동 방지
        if target == "대전선병원":
            candidates = df[
                df["_norm_name"].str.contains("대전선병원", na=False)
                | (
                    df["_norm_name"].str.contains("선병원", na=False)
                    & df.astype(str)
                    .apply(lambda col: col.str.contains("목중로", na=False))
                    .any(axis=1)
                )
            ].copy()

        if target == "유성선병원":
            candidates = df[
                df["_norm_name"].str.contains("유성선병원", na=False)
                | (
                    df["_norm_name"].str.contains("선병원", na=False)
                    & df.astype(str)
                    .apply(lambda col: col.str.contains("북유성대로", na=False))
                    .any(axis=1)
                )
            ].copy()

        if len(candidates) == 0:
            matched_rows.append({
                "target_hospital": target,
                "match_status": "미매칭",
            })
        else:
            row = candidates.iloc[0].to_dict()
            row["target_hospital"] = target
            row["match_status"] = "OK"
            row["api_matched_name"] = row.get(name_col)
            matched_rows.append(row)

    return pd.DataFrame(matched_rows)


def build_realtime_update(stage1=DEFAULT_STAGE1):
    """
    실시간 업데이트용 함수.

    현재는 실시간 가용병상정보 API만 호출한다.
    병원 기본정보는 이미 구축한 DB를 사용하므로,
    getEgytListInfoInqire는 호출하지 않는다.
    """
    realtime_raw = fetch_realtime_beds(stage1=stage1)
    realtime_10 = match_target_hospitals(realtime_raw)

    # 목록정보 API는 500 에러가 날 수 있고,
    # 실시간 갱신에는 필수 정보가 아니므로 빈 데이터프레임 처리.
    list_raw = pd.DataFrame()
    list_10 = pd.DataFrame()

    return realtime_10, list_10, realtime_raw, list_raw
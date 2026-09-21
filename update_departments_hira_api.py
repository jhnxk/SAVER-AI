import os
import re
import time
import math
import shutil
import requests
import pandas as pd
import xml.etree.ElementTree as ET

from pathlib import Path
from datetime import datetime
from urllib.parse import urlencode
from dotenv import load_dotenv


# =========================================================
# 1. 기본 설정
# =========================================================

load_dotenv()

HIRA_DETAIL_SERVICE_KEY = (
    os.getenv("HIRA_DETAIL_SERVICE_KEY")
    or os.getenv("HIRA_DETAILED_SERVICE_KEY")
    or os.getenv("HIRA_SERVICE_KEY")
)

if not HIRA_DETAIL_SERVICE_KEY:
    raise RuntimeError(
        ".env 파일에 HIRA_DETAIL_SERVICE_KEY가 없습니다.\n"
        "예: HIRA_DETAIL_SERVICE_KEY=의료기관별상세정보서비스_API_인증키"
    )

OUTPUT_FILE = "national_hospital_db_departments_hira_updated.xlsx"
BACKUP_DIR = Path("hira_backups")

INPUT_CANDIDATES = [
    OUTPUT_FILE,
    "national_hospital_db_realtime_updated.xlsx",
    "national_hospital_master.xlsx",
]

# 0이면 전체 병원 업데이트, 5면 앞 5개만 테스트
DEPARTMENT_UPDATE_LIMIT = int(os.getenv("DEPARTMENT_UPDATE_LIMIT", "5"))
SLEEP_BETWEEN_CALLS = float(os.getenv("HIRA_DEPARTMENT_SLEEP", "0.2"))
NUM_OF_ROWS = 200

BASE_URL = "https://apis.data.go.kr/B551182/MadmDtlInfoService2.8"

ENDPOINTS = {
    "departments": {
        "path": "/getDgsbjtInfo2.8",
        "desc": "진료과목정보",
    },
    "specialist_count": {
        "path": "/getSpcSbjtSdrInfo2.8",
        "desc": "전문과목별 전문의 수",
    },
    "special_diag": {
        "path": "/getSpclDiagInfo2.8",
        "desc": "특수진료정보",
    },
    "equipment": {
        "path": "/getMedOftInfo2.8",
        "desc": "의료장비정보",
    },
}

# 최종 메인 DB에 남길 기본 보완 컬럼
BASE_EXTRA_COLUMNS = [
    "departments",
    "departments_last_checked_at",
    "departments_error",
    "specialist_count_error",
    "special_diag_summary",
    "special_diag_error",
    "medical_equipment_summary",
    "medical_equipment_error",
]

BOOL_TEXT_COLUMNS = [
    "has_emergency_medicine",
    "has_neurology",
    "has_neurosurgery",
    "has_cardiology",
    "has_cardiovascular_medicine",
    "has_general_surgery",
    "has_thoracic_surgery",
    "has_orthopedics",
    "has_obgyn",
    "has_pediatrics",
    "has_neonatology",
]

# 사용자가 빼길 원한 메타/중복 컬럼
DROP_FROM_FINAL_DB = [
    "departments_confidence",
    "departments_update_method",
    "departments_evidence_summary",
    "specialist_count_summary",
    "departments_source_url",
    "specialist_count_source_url",
    "special_diag_source_url",
    "medical_equipment_source_url",
]

# 예전 버전에서 만들었던 '주요 과만 따로 뽑은' 컬럼들. 최종 DB에서는 제거한다.
OLD_CORE_SPECIALIST_COLUMNS = [
    "specialist_emergency_medicine_count",
    "specialist_neurology_count",
    "specialist_neurosurgery_count",
    "specialist_cardiology_count",
    "specialist_general_surgery_count",
    "specialist_thoracic_surgery_count",
    "specialist_orthopedics_count",
    "specialist_obgyn_count",
    "specialist_pediatrics_count",
]


# =========================================================
# 2. 파일 읽기
# =========================================================

def find_input_file():
    for file in INPUT_CANDIDATES:
        if Path(file).exists():
            return file
    raise FileNotFoundError(
        "입력 파일이 없습니다.\n"
        "- national_hospital_db_realtime_updated.xlsx\n"
        "- national_hospital_master.xlsx\n"
        "중 하나가 필요합니다."
    )


def choose_sheet_name(excel_file):
    xls = pd.ExcelFile(excel_file)
    preferred_sheets = [
        "national_hospital_db",
        "hospital_master_realtime",
        "hospital_db_realtime",
        "hospital_master",
        "national_hospital_master",
    ]
    for sheet in preferred_sheets:
        if sheet in xls.sheet_names:
            return sheet
    return xls.sheet_names[0]


def load_input_dataframe():
    input_file = find_input_file()
    sheet_name = choose_sheet_name(input_file)

    print(f"[1/6] 입력 파일 읽는 중: {input_file}")
    print(f"    - 사용 시트: {sheet_name}")

    df = pd.read_excel(input_file, sheet_name=sheet_name)
    df = df.drop(columns=[col for col in df.columns if str(col).startswith("Unnamed")], errors="ignore")

    # 기존 HIRA 최종본을 다시 수집할 때는 이미 수집된 과별 전문의 수를 보존한다.
    # 각 병원의 전문의 API가 정상 응답한 경우에만 해당 행의 값을 새 결과로 갱신한다.
    df = df.drop(columns=OLD_CORE_SPECIALIST_COLUMNS + DROP_FROM_FINAL_DB, errors="ignore")

    for col in BASE_EXTRA_COLUMNS + BOOL_TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("object").fillna("")

    return df, input_file, sheet_name


def backup_existing_output():
    output_path = Path(OUTPUT_FILE)

    if not output_path.exists():
        return None

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"national_hospital_db_departments_hira_updated_{timestamp}.xlsx"
    shutil.copy2(output_path, backup_path)

    print(f"기존 HIRA 최종본 백업 완료: {backup_path}")
    return backup_path


# =========================================================
# 3. 공통 유틸
# =========================================================

def pick_first_existing(row, candidates):
    for col in candidates:
        if col in row:
            value = row.get(col)
            if pd.notna(value) and str(value).strip() != "":
                return str(value).strip()
    return ""


def get_hospital_name(row):
    return pick_first_existing(row, ["hospital_name", "target_hospital", "dutyname", "dutyName", "yadmNm", "name"])


def get_hospital_address(row):
    return pick_first_existing(row, ["address", "dutyAddr", "dutyaddr", "addr"])


def get_ykiho(row):
    return pick_first_existing(row, ["encrypted_ykiho", "ykiho", "YKIHO", "encryptedYkiho"])


def get_text(elem, tag):
    if elem is None:
        return ""
    found = elem.find(tag)
    if found is None or found.text is None:
        return ""
    return found.text.strip()


def parse_xml_items(xml_text):
    root = ET.fromstring(xml_text)
    header = root.find(".//header")
    body = root.find(".//body")

    result_code = get_text(header, "resultCode")
    result_msg = get_text(header, "resultMsg")
    total_count_text = get_text(body, "totalCount")

    try:
        total_count = int(total_count_text)
    except Exception:
        total_count = 0

    rows = []
    for item in root.findall(".//item"):
        row = {}
        for child in list(item):
            row[child.tag] = child.text.strip() if child.text else ""
        rows.append(row)

    return result_code, result_msg, total_count, pd.DataFrame(rows)


def build_url(endpoint_path, params):
    query = urlencode(params)
    return f"{BASE_URL}{endpoint_path}?serviceKey={HIRA_DETAIL_SERVICE_KEY}&{query}"


def call_api(endpoint_key, ykiho, page_no=1):
    endpoint = ENDPOINTS[endpoint_key]
    params = {"pageNo": str(page_no), "numOfRows": str(NUM_OF_ROWS), "ykiho": ykiho}
    url = build_url(endpoint["path"], params)

    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code} / {endpoint['desc']} 호출 실패 / 응답: {response.text[:500]}")

    code, msg, total_count, first_df = parse_xml_items(response.text)
    if code and code not in ["00", "0000", "NORMAL SERVICE."]:
        raise RuntimeError(f"API resultCode={code}, resultMsg={msg}, endpoint={endpoint['desc']}")

    all_dfs = [first_df]
    if total_count > NUM_OF_ROWS:
        total_pages = math.ceil(total_count / NUM_OF_ROWS)
        for p in range(2, total_pages + 1):
            page_url = build_url(endpoint["path"], {"pageNo": str(p), "numOfRows": str(NUM_OF_ROWS), "ykiho": ykiho})
            page_response = requests.get(page_url, timeout=60)
            if page_response.status_code != 200:
                raise RuntimeError(f"HTTP {page_response.status_code} / page={p} / 응답: {page_response.text[:500]}")
            _, _, _, page_df = parse_xml_items(page_response.text)
            all_dfs.append(page_df)
            time.sleep(0.1)

    result_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return {"endpoint_key": endpoint_key, "desc": endpoint["desc"], "source_url": BASE_URL + endpoint["path"], "total_count": total_count, "df": result_df}


# =========================================================
# 4. 응답 데이터 추출
# =========================================================

def find_best_text_column(df, candidate_cols, keyword_hint=None):
    if df is None or df.empty:
        return None
    for col in candidate_cols:
        if col in df.columns:
            return col
    if keyword_hint:
        best_col = None
        best_score = 0
        for col in df.columns:
            values = df[col].dropna().astype(str).tolist()
            score = sum(1 for v in values if any(k in v for k in keyword_hint))
            if score > best_score:
                best_score = score
                best_col = col
        return best_col
    return None


def extract_department_names(dept_df):
    col = find_best_text_column(
        dept_df,
        ["dgsbjtCdNm", "dgsbjtCdName", "dgsbjtNm", "mclCdNm", "deptNm", "subjectNm", "clCdNm"],
        keyword_hint=["과", "내과", "외과", "의학"],
    )
    if col is None:
        return []

    departments = []
    for value in dept_df[col].dropna().astype(str):
        value = value.strip()
        if value and value not in departments:
            departments.append(value)
    return departments


def safe_int(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    value = str(value).strip()
    if value == "":
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def sanitize_column_part(text):
    text = str(text).strip()
    text = re.sub(r"[^0-9A-Za-z가-힣_]", "", text)
    return text


def build_specialist_dynamic_column(dept_code, dept_name):
    code = sanitize_column_part(dept_code)
    name = sanitize_column_part(dept_name)
    if code:
        return f"specialist_count_{code}_{name}"
    return f"specialist_count_{name}"


def extract_all_specialist_counts(spc_df):
    """
    HIRA /getSpcSbjtSdrInfo2.8 응답의 모든 과별 전문의 수를 수집한다.
    실제 확인된 핵심 컬럼: dgsbjtCd, dgsbjtCdNm, dtlSdrCnt
    """
    dynamic_counts = {}
    long_rows = []

    if spc_df is None or spc_df.empty:
        return dynamic_counts, long_rows

    subject_col = find_best_text_column(
        spc_df,
        ["dgsbjtCdNm", "dgsbjtNm", "spcSbjtCdNm", "mclCdNm", "deptNm", "subjectNm"],
        keyword_hint=["과", "내과", "외과", "의학"],
    )

    code_col = None
    for col in ["dgsbjtCd", "spcSbjtCd", "mclCd", "deptCd", "subjectCd"]:
        if col in spc_df.columns:
            code_col = col
            break

    count_col = None
    for col in ["dtlSdrCnt", "sdrCnt", "spclSdrCnt", "spcSdrCnt", "spdrCnt", "spcDrCnt", "drTotCnt", "chosCnt", "spclCnt", "specialistCnt", "drCnt", "doctorCnt", "cnt"]:
        if col in spc_df.columns:
            count_col = col
            break

    if subject_col is None or count_col is None:
        return dynamic_counts, long_rows

    for _, row in spc_df.iterrows():
        dept_name = str(row.get(subject_col, "")).strip()
        dept_code = str(row.get(code_col, "")).strip() if code_col else ""
        count_value = safe_int(row.get(count_col, ""))

        if not dept_name or count_value is None:
            continue

        dynamic_col = build_specialist_dynamic_column(dept_code, dept_name)
        dynamic_counts[dynamic_col] = count_value
        long_rows.append({"dgsbjtCd": dept_code, "dgsbjtCdNm": dept_name, "dtlSdrCnt": count_value, "dynamic_column": dynamic_col})

    return dynamic_counts, long_rows


def extract_simple_summary(df):
    if df is None or df.empty:
        return ""
    parts = []
    for _, row in df.head(30).iterrows():
        values = []
        for value in row.to_dict().values():
            if pd.notna(value) and str(value).strip():
                values.append(str(value).strip())
        text = " / ".join(values)
        if text and text not in parts:
            parts.append(text)
    return " || ".join(parts)


def has_any_department(departments, keywords):
    text = " ".join(departments)
    return "Y" if any(keyword in text for keyword in keywords) else "N"


def build_department_flags(departments):
    return {
        "has_emergency_medicine": has_any_department(departments, ["응급의학과"]),
        "has_neurology": has_any_department(departments, ["신경과"]),
        "has_neurosurgery": has_any_department(departments, ["신경외과"]),
        "has_cardiology": has_any_department(departments, ["심장내과", "순환기내과"]),
        "has_cardiovascular_medicine": has_any_department(departments, ["심장내과", "순환기내과", "심혈관"]),
        "has_general_surgery": has_any_department(departments, ["외과"]),
        "has_thoracic_surgery": has_any_department(departments, ["흉부외과", "심장혈관흉부외과"]),
        "has_orthopedics": has_any_department(departments, ["정형외과"]),
        "has_obgyn": has_any_department(departments, ["산부인과"]),
        "has_pediatrics": has_any_department(departments, ["소아청소년과", "소아과"]),
        "has_neonatology": has_any_department(departments, ["신생아", "신생아과"]),
    }


# =========================================================
# 5. 병원별 업데이트
# =========================================================

def select_update_targets(df):
    target_indices = []
    for idx, row in df.iterrows():
        hospital_name = get_hospital_name(row)
        ykiho = get_ykiho(row)
        if hospital_name and ykiho:
            target_indices.append(idx)

    if DEPARTMENT_UPDATE_LIMIT > 0:
        target_indices = target_indices[:DEPARTMENT_UPDATE_LIMIT]

    print(f"[2/6] 업데이트 대상 병원 수: {len(target_indices)}")
    print(f"    - limit: {DEPARTMENT_UPDATE_LIMIT}")
    return target_indices


def clear_specialist_counts_for_row(df, idx):
    dynamic_cols = [
        col for col in df.columns
        if re.match(r"^specialist_count_\d+_", str(col))
    ]

    for col in dynamic_cols:
        df.at[idx, col] = pd.NA


def update_one_hospital(df, idx, specialist_long_rows):
    row = df.loc[idx]
    hospital_name = get_hospital_name(row)
    address = get_hospital_address(row)
    ykiho = get_ykiho(row)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"  - {hospital_name} / {address}")

    # 1. 진료과목정보
    try:
        dept_result = call_api("departments", ykiho)
        departments = extract_department_names(dept_result["df"])

        df.at[idx, "departments_last_checked_at"] = now

        if departments:
            flags = build_department_flags(departments)

            df.at[idx, "departments"] = ", ".join(departments)
            df.at[idx, "departments_error"] = ""

            for col, value in flags.items():
                df.at[idx, col] = value

            print(f"    진료과목정보 성공: {len(departments)}개")
        else:
            df.at[idx, "departments_error"] = "빈 응답 - 기존 값 보존"
            print("    진료과목정보 빈 응답: 기존 값 보존")
    except Exception as e:
        df.at[idx, "departments_last_checked_at"] = now
        df.at[idx, "departments_error"] = str(e)
        print(f"    진료과목정보 실패: {e}")

    time.sleep(SLEEP_BETWEEN_CALLS)

    # 2. 전문과목별 전문의 수: 해당 병원에 존재하는 모든 과를 컬럼으로 생성
    try:
        spc_result = call_api("specialist_count", ykiho)
        dynamic_counts, long_rows = extract_all_specialist_counts(spc_result["df"])

        if dynamic_counts:
            # 정상 응답한 병원만 기존 과별 전문의 수를 새 결과로 교체한다.
            # 빈 응답/오류일 때는 기존 정상값을 지우지 않는다.
            clear_specialist_counts_for_row(df, idx)

            for col, value in dynamic_counts.items():
                if col not in df.columns:
                    df[col] = pd.NA
                df.at[idx, col] = value

            for long_row in long_rows:
                specialist_long_rows.append(
                    {
                        "hospital_name": hospital_name,
                        "address": address,
                        "encrypted_ykiho": ykiho,
                        "dgsbjtCd": long_row.get("dgsbjtCd", ""),
                        "dgsbjtCdNm": long_row.get("dgsbjtCdNm", ""),
                        "dtlSdrCnt": long_row.get("dtlSdrCnt", ""),
                        "dynamic_column": long_row.get("dynamic_column", ""),
                        "source_url": spc_result["source_url"],
                        "fetched_at": now,
                    }
                )

            df.at[idx, "specialist_count_error"] = ""
            print(f"    전문의 수 성공: {len(dynamic_counts)}개 과")
        else:
            df.at[idx, "specialist_count_error"] = "빈 응답 - 기존 값 보존"
            print("    전문의 수 빈 응답: 기존 값 보존")
    except Exception as e:
        df.at[idx, "specialist_count_error"] = str(e)
        print(f"    전문의 수 실패: {e}")

    time.sleep(SLEEP_BETWEEN_CALLS)

    # 3. 특수진료정보
    try:
        special_result = call_api("special_diag", ykiho)
        summary = extract_simple_summary(special_result["df"])

        if summary:
            df.at[idx, "special_diag_summary"] = summary
            df.at[idx, "special_diag_error"] = ""
            print("    특수진료정보 성공")
        else:
            df.at[idx, "special_diag_error"] = "빈 응답 - 기존 값 보존"
            print("    특수진료정보 빈 응답: 기존 값 보존")
    except Exception as e:
        df.at[idx, "special_diag_error"] = str(e)
        print(f"    특수진료정보 실패: {e}")

    time.sleep(SLEEP_BETWEEN_CALLS)

    # 4. 의료장비정보
    try:
        equip_result = call_api("equipment", ykiho)
        summary = extract_simple_summary(equip_result["df"])

        if summary:
            df.at[idx, "medical_equipment_summary"] = summary
            df.at[idx, "medical_equipment_error"] = ""
            print("    의료장비정보 성공")
        else:
            df.at[idx, "medical_equipment_error"] = "빈 응답 - 기존 값 보존"
            print("    의료장비정보 빈 응답: 기존 값 보존")
    except Exception as e:
        df.at[idx, "medical_equipment_error"] = str(e)
        print(f"    의료장비정보 실패: {e}")

    return df


# =========================================================
# 6. 저장
# =========================================================

def is_dynamic_specialist_col(col):
    return bool(re.match(r"^specialist_count_\d+_", str(col)))


def sort_specialist_columns(cols):
    def key_func(col):
        m = re.match(r"^specialist_count_(\d+)_", str(col))
        code = int(m.group(1)) if m else 9999
        return (code, str(col))
    return sorted(cols, key=key_func)


def save_output(df, input_file, sheet_name, specialist_long_rows):
    print("[6/6] 결과 저장 중...")

    dynamic_specialist_cols = sort_specialist_columns([col for col in df.columns if is_dynamic_specialist_col(col)])

    for col in dynamic_specialist_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 요청한 불필요 컬럼 제거 + 예전 주요 과 전용 컬럼 제거
    df = df.drop(columns=DROP_FROM_FINAL_DB + OLD_CORE_SPECIALIST_COLUMNS, errors="ignore")

    # 컬럼 순서 정리: 기본 정보 → 실시간 응급자원 → 진료과 → 모든 과별 전문의 수 → 특수진료/장비
    first_cols = [
        "hospital_name", "target_hospital", "hospital_type", "sido", "sigungu", "address", "main_tel",
        "hpid", "realtime_match_status", "match_method", "match_score", "api_matched_name",
        "hvidate", "hvec", "hvoc", "hvgc", "hvicc", "hvncc", "hvcc", "hvccc", "hv2", "hv3", "hv4", "hv5", "hv6",
        "hvctayn", "hvmriayn", "hvangioayn", "hvventiayn", "data_fetched_at",
        "departments", "departments_last_checked_at",
        "has_emergency_medicine", "has_neurology", "has_neurosurgery", "has_cardiology", "has_cardiovascular_medicine",
        "has_general_surgery", "has_thoracic_surgery", "has_orthopedics", "has_obgyn", "has_pediatrics", "has_neonatology",
    ]
    tail_cols = ["special_diag_summary", "medical_equipment_summary", "departments_error", "specialist_count_error", "special_diag_error", "medical_equipment_error"]
    ordered_cols = [c for c in first_cols if c in df.columns] + dynamic_specialist_cols + [c for c in tail_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + other_cols]

    filled_departments_count = int(df["departments"].astype(str).str.strip().ne("").sum()) if "departments" in df.columns else 0
    specialist_hospital_count = int(df[dynamic_specialist_cols].notna().any(axis=1).sum()) if dynamic_specialist_cols else 0
    special_diag_filled_count = int(df["special_diag_summary"].astype(str).str.strip().ne("").sum()) if "special_diag_summary" in df.columns else 0
    equipment_filled_count = int(df["medical_equipment_summary"].astype(str).str.strip().ne("").sum()) if "medical_equipment_summary" in df.columns else 0

    specialist_column_meta = []
    for col in dynamic_specialist_cols:
        specialist_column_meta.append({"column": col, "filled_hospital_count": int(df[col].notna().sum())})

    specialist_meta_df = pd.DataFrame(specialist_column_meta)
    specialist_long_df = pd.DataFrame(specialist_long_rows)
    if not specialist_long_df.empty:
        specialist_long_df = specialist_long_df.sort_values(by=["hospital_name", "dgsbjtCd"]).reset_index(drop=True)

    meta = pd.DataFrame(
        [
            {
                "input_file": input_file,
                "input_sheet": sheet_name,
                "output_file": OUTPUT_FILE,
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "department_update_limit": DEPARTMENT_UPDATE_LIMIT,
                "filled_departments_count": filled_departments_count,
                "specialist_hospital_count": specialist_hospital_count,
                "specialist_dynamic_column_count": len(dynamic_specialist_cols),
                "specialist_long_row_count": len(specialist_long_df),
                "filled_special_diag_summary": special_diag_filled_count,
                "filled_medical_equipment_summary": equipment_filled_count,
                "note": "전문의 수는 HIRA getSpcSbjtSdrInfo2.8의 dgsbjtCd, dgsbjtCdNm, dtlSdrCnt를 사용해 API에 존재하는 모든 과별 전문의 수를 specialist_count_코드_진료과명 컬럼으로 저장했다.",
            }
        ]
    )

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="national_hospital_db")
        specialist_long_df.to_excel(writer, index=False, sheet_name="specialist_counts_long")
        specialist_meta_df.to_excel(writer, index=False, sheet_name="specialist_column_meta")
        meta.to_excel(writer, index=False, sheet_name="update_meta")

    print(f"저장 완료: {OUTPUT_FILE}")
    print(f"- 진료과 채워진 병원 수: {filled_departments_count}개")
    print(f"- 전문의 수가 있는 병원 수: {specialist_hospital_count}개")
    print(f"- 생성된 전체 과별 전문의 수 컬럼 수: {len(dynamic_specialist_cols)}개")
    print(f"- 전문의 수 long table 행 수: {len(specialist_long_df)}개")
    print(f"- 특수진료정보 병원 수: {special_diag_filled_count}개")
    print(f"- 의료장비정보 병원 수: {equipment_filled_count}개")
    if dynamic_specialist_cols:
        print("- 생성 컬럼 예시:")
        for col in dynamic_specialist_cols[:30]:
            print(f"  {col}")


# =========================================================
# 7. 메인
# =========================================================

def main():
    print("==============================================")
    print("HIRA 상세정보 API 기반 병원 정보 보완 시작")
    print("==============================================")

    backup_existing_output()
    df, input_file, sheet_name = load_input_dataframe()
    target_indices = select_update_targets(df)
    specialist_long_rows = []

    if not target_indices:
        print("업데이트 대상이 없습니다.")
        save_output(df, input_file, sheet_name, specialist_long_rows)
        return

    print("[3/6] 병원별 HIRA 상세정보 API 호출 중...")
    for order, idx in enumerate(target_indices, start=1):
        print(f"[{order}/{len(target_indices)}]")
        df = update_one_hospital(df, idx, specialist_long_rows)

    print("[4/6] 업데이트 완료")
    print("[5/6] 저장 준비")
    save_output(df, input_file, sheet_name, specialist_long_rows)

    print("==============================================")
    print("작업 완료")
    print("==============================================")


if __name__ == "__main__":
    main()

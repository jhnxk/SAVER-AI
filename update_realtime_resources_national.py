import os
import re
import time
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from pathlib import Path
from datetime import datetime
from importlib.util import find_spec

from api_client import fetch_realtime_beds


# =========================================================
# 1. 기본 설정
# =========================================================

INPUT_FILE = "national_hospital_master.xlsx"
OUTPUT_FILE = "national_hospital_db_realtime_updated.xlsx"

# 전국 시도명
SIDO_LIST = [
    "서울특별시",
    "부산광역시",
    "대구광역시",
    "인천광역시",
    "대전광역시",
    "울산광역시",
    "세종특별자치시",
    "경기도",
    "강원특별자치도",
    "충청북도",
    "충청남도",
    "전북특별자치도",
    "경상북도",
    "경상남도",
    "제주특별자치도",
    # 옛 광주·전남 관할의 실시간 데이터는 통합 지역명으로 조회한다.
    "전남광주통합특별시",
]

# 화면/알고리즘에서 사용할 실시간 컬럼
REALTIME_COLUMNS = [
    "hpid",
    "hvidate",
    "hvec",
    "hvoc",
    "hvgc",
    "hvicc",
    "hvncc",
    "hvcc",
    "hvccc",
    "hv2",
    "hv3",
    "hv4",
    "hv5",
    "hv6",
    "hvctayn",
    "hvmriayn",
    "hvangioayn",
    "hvventiayn",
    "data_fetched_at",
]

# API에서 병원명으로 쓰일 수 있는 컬럼 후보
API_NAME_COLUMNS = [
    "dutyName",
    "dutyname",
    "dutyNm",
    "dutyNm",
    "yadmNm",
]

# API에서 주소로 쓰일 수 있는 컬럼 후보
API_ADDRESS_COLUMNS = [
    "dutyAddr",
    "dutyaddr",
    "addr",
    "address",
]

# 마스터 DB 병원명 컬럼 후보
MASTER_NAME_COLUMNS = [
    "hospital_name",
    "target_hospital",
    "dutyname",
    "dutyName",
    "yadmNm",
]

# 마스터 DB 주소 컬럼 후보
MASTER_ADDRESS_COLUMNS = [
    "address",
    "dutyAddr",
    "dutyaddr",
    "addr",
]

# fuzzy matching 기준
FUZZY_THRESHOLD = 0.82


# =========================================================
# 2. 유틸 함수
# =========================================================

def pick_first_existing(row, candidates):
    for col in candidates:
        if col in row:
            value = row.get(col)
            if pd.notna(value) and str(value).strip() != "":
                return str(value).strip()
    return ""


def normalize_text(value):
    """
    병원명/주소 비교용 정규화.
    """
    if value is None or pd.isna(value):
        return ""

    s = str(value).strip()

    remove_tokens = [
        "의료법인",
        "학교법인",
        "재단법인",
        "사회복지법인",
        "사단법인",
        "공익재단법인",
        "근로복지공단",
        "국민건강보험공단",
        "한국보훈복지의료공단",
        "가톨릭대학교",
        "학교법인가톨릭학원",
        "학교법인연세대학교",
        "학교법인고려중앙학원",
        "학교법인동은학원",
        "학교법인인제학원",
        "학교법인울산공업학원",
        "의료재단",
        "영훈의료재단",
        "대학교의료원",
        "의료원",
        "부속",
        "부설",
        "진료소",
        "권역응급의료센터",
        "지역응급의료센터",
        "지역응급의료기관",
        "응급의료센터",
        "응급실",
        "상급종합병원",
        "종합병원",
    ]

    for token in remove_tokens:
        s = s.replace(token, "")

    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = re.sub(r"[^0-9A-Za-z가-힣]", "", s)

    return s.lower()


def normalize_hospital_name(value):
    s = normalize_text(value)

    # 자주 생기는 표기 차이 보정
    replacements = {
        "대학병원": "대학교병원",
        "부속병원": "병원",
        "서울아산": "아산",
        "강릉아산": "강릉아산병원",
    }

    for old, new in replacements.items():
        s = s.replace(normalize_text(old), normalize_text(new))

    return s


def normalize_sido(value):
    """
    마스터 DB의 sido가 '강원', API가 '강원특별자치도'처럼 다를 수 있어 비교용으로 정리.
    """
    if value is None or pd.isna(value):
        return ""

    s = str(value).strip()

    mapping = {
        "서울특별시": "서울",
        "부산광역시": "부산",
        "대구광역시": "대구",
        "인천광역시": "인천",
        "광주광역시": "광주",
        "대전광역시": "대전",
        "울산광역시": "울산",
        "세종특별자치시": "세종",
        "경기도": "경기",
        "강원특별자치도": "강원",
        "강원도": "강원",
        "충청북도": "충북",
        "충청남도": "충남",
        "전북특별자치도": "전북",
        "전라북도": "전북",
        "전라남도": "전남",
        "경상북도": "경북",
        "경상남도": "경남",
        "제주특별자치도": "제주",
        "전남광주통합특별시": "전남광주",
    }

    return mapping.get(s, s)


def extract_api_name(row):
    return pick_first_existing(row, API_NAME_COLUMNS)


def extract_api_address(row):
    return pick_first_existing(row, API_ADDRESS_COLUMNS)


def extract_master_name(row):
    return pick_first_existing(row, MASTER_NAME_COLUMNS)


def extract_master_address(row):
    return pick_first_existing(row, MASTER_ADDRESS_COLUMNS)


def get_api_name_col(df):
    for col in API_NAME_COLUMNS:
        if col in df.columns:
            return col
    return None


def clean_yn_value(value):
    """
    Y/N 컬럼에서 N1 같은 값 정리.
    """
    if value is None or pd.isna(value):
        return ""

    s = str(value).strip().upper()

    if s.startswith("Y"):
        return "Y"

    if s.startswith("N"):
        return "N"

    return s


def prepare_realtime_df(realtime_df):
    """
    API 실시간 데이터에 매칭용 보조 컬럼 추가.
    """
    df = realtime_df.copy()

    api_name_col = get_api_name_col(df)

    if api_name_col is None:
        df["_api_name"] = ""
    else:
        df["_api_name"] = df[api_name_col].astype(str)

    df["_api_norm_name"] = df["_api_name"].apply(normalize_hospital_name)

    address_col = None
    for col in API_ADDRESS_COLUMNS:
        if col in df.columns:
            address_col = col
            break

    if address_col:
        df["_api_address"] = df[address_col].astype(str)
        df["_api_norm_address"] = df["_api_address"].apply(normalize_text)
    else:
        df["_api_address"] = ""
        df["_api_norm_address"] = ""

    if "hpid" not in df.columns:
        df["hpid"] = ""

    df["_api_hpid"] = df["hpid"].astype(str).str.strip()

    for col in ["hvctayn", "hvmriayn", "hvangioayn", "hvventiayn"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_yn_value)

    return df


def prepare_master_df(master_df):
    """
    마스터 DB에 매칭용 보조 컬럼 추가.
    """
    df = master_df.copy()

    if "hospital_name" not in df.columns:
        df["hospital_name"] = df.apply(lambda row: extract_master_name(row), axis=1)

    if "target_hospital" not in df.columns:
        df["target_hospital"] = df["hospital_name"]

    if "address" not in df.columns:
        df["address"] = df.apply(lambda row: extract_master_address(row), axis=1)

    if "sido" not in df.columns:
        df["sido"] = ""

    if "sigungu" not in df.columns:
        df["sigungu"] = ""

    if "hpid" not in df.columns:
        df["hpid"] = ""

    df["_master_name"] = df.apply(lambda row: extract_master_name(row), axis=1)
    df["_master_address"] = df.apply(lambda row: extract_master_address(row), axis=1)

    df["_master_norm_name"] = df["_master_name"].apply(normalize_hospital_name)
    df["_master_norm_address"] = df["_master_address"].apply(normalize_text)
    df["_master_hpid"] = df["hpid"].astype(str).str.strip()
    df["_master_sido_short"] = df["sido"].apply(normalize_sido)

    return df


def copy_realtime_values(master_row, api_row, match_status, match_method, match_score, match_note):
    """
    마스터 행에 API 실시간 값을 붙인다.
    """
    out = master_row.copy()

    for col in REALTIME_COLUMNS:
        if col in api_row:
            out[col] = api_row.get(col)
        elif col not in out:
            out[col] = ""

    out["realtime_match_status"] = match_status
    out["match_method"] = match_method
    out["match_score"] = match_score
    out["match_note"] = match_note
    out["api_matched_name"] = api_row.get("_api_name", "")

    return out


def mark_unmatched(master_row, note):
    out = master_row.copy()

    for col in REALTIME_COLUMNS:
        if col not in out:
            out[col] = ""

    out["realtime_match_status"] = "응급의료API_미매칭"
    out["match_method"] = "unmatched"
    out["match_score"] = 0
    out["match_note"] = note
    out["api_matched_name"] = ""

    return out


# =========================================================
# 3. 전국 실시간 API 수집
# =========================================================

def fetch_national_realtime_api():
    """시도별 실시간 API를 병렬로 호출해 전국 데이터를 합친다.

    기존에는 최대 18개 시·도를 순차 호출하여 한 지역의 응답 지연/timeout이
    전체 갱신시간에 누적됐다. API 내용과 매칭 로직은 바꾸지 않고 독립적인
    시·도 요청만 최대 4개씩 병렬 실행한다.
    """
    print("[2/4] 전국 응급의료기관 실시간 API 호출 중 (최대 4개 병렬)...")

    all_dfs = []
    request_seconds = {}

    def fetch_one(index, sido):
        started_at = time.perf_counter()
        try:
            df = fetch_realtime_beds(stage1=sido)
        finally:
            request_seconds[index] = time.perf_counter() - started_at
        if df is not None and not df.empty:
            df = df.copy()
            df["api_query_sido"] = sido
        return index, sido, df

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(fetch_one, idx, sido): (idx, sido)
            for idx, sido in enumerate(SIDO_LIST, start=1)
        }
        for future in as_completed(futures):
            idx, sido = futures[future]
            try:
                _, _, df = future.result()
                elapsed = request_seconds[idx]
                if df is not None and not df.empty:
                    all_dfs.append(df)
                    print(f"  - [{idx}/{len(SIDO_LIST)}] {sido}: {len(df)}건 ({elapsed:.2f}초)")
                else:
                    print(f"  - [{idx}/{len(SIDO_LIST)}] {sido}: 0건 ({elapsed:.2f}초)")
            except Exception as e:
                elapsed = request_seconds.get(idx, 0)
                print(f"  - [{idx}/{len(SIDO_LIST)}] {sido} 실패 ({elapsed:.2f}초): {type(e).__name__}")

    if not all_dfs:
        print("    ! 실시간 API에서 수집된 데이터가 없습니다.")
        return pd.DataFrame()

    realtime_df = pd.concat(all_dfs, ignore_index=True)
    realtime_df = realtime_df.drop_duplicates()
    print(f"    - 전국 실시간 API 수집 완료: {len(realtime_df)}건")
    return realtime_df



# =========================================================
# 4. 매칭 로직
# =========================================================

def filter_by_region_candidates(master_row, search_columns, all_positions):
    """Filter row positions in the original order, retaining the prior nonempty fallback."""
    master_sido = str(master_row.get("_master_sido_short", "")).strip()
    master_addr = str(master_row.get("_master_norm_address", "")).strip()
    master_sigungu = str(master_row.get("sigungu", "")).strip()

    addresses = search_columns["_api_address"]
    positions = all_positions
    if master_sido:
        pattern = re.compile(master_sido)
        regions = search_columns["api_query_sido"]
        region_positions = [
            pos for pos in positions
            if regions[pos] == master_sido or pattern.search(addresses[pos])
        ]
        positions = region_positions or positions

    if master_sigungu:
        pattern = re.compile(master_sigungu)
        district_positions = [pos for pos in positions if pattern.search(addresses[pos])]
        positions = district_positions or positions

    if master_addr:
        pattern = re.compile(master_addr[:6])
        normalized_addresses = search_columns["_api_norm_address"]
        address_positions = [
            pos for pos in positions if pattern.search(normalized_addresses[pos])
        ]
        positions = address_positions or positions

    return positions


def find_best_match(master_row, realtime_df, search_columns, all_positions):
    """
    한 개 마스터 병원에 대해 가장 적절한 API 행을 찾는다.
    매칭 순서:
    1. hpid 정확 매칭
    2. 정규화 병원명 완전 일치
    3. 정규화 병원명 포함 관계
    4. 지역 후보 내 fuzzy matching
    """

    master_hpid = str(master_row.get("_master_hpid", "")).strip()
    master_name = str(master_row.get("_master_name", "")).strip()
    master_norm = str(master_row.get("_master_norm_name", "")).strip()

    if master_norm == "":
        return None, "unmatched", 0, "마스터 병원명 없음"

    # 1. hpid 정확 매칭
    if master_hpid and master_hpid.lower() not in ["nan", "none"]:
        hpid_matches = realtime_df["_api_hpid"].eq(master_hpid).to_numpy().nonzero()[0]
        if hpid_matches.size:
            api_row = realtime_df.iloc[hpid_matches[0]].to_dict()
            return api_row, "hpid_exact", 100, "hpid 정확 매칭"

    positions = filter_by_region_candidates(master_row, search_columns, all_positions)
    names = search_columns["_api_norm_name"]

    # 2. 병원명 완전 일치
    for pos in positions:
        if names[pos] == master_norm:
            api_row = realtime_df.iloc[pos].to_dict()
            return api_row, "name_exact", 95, "정규화 병원명 완전 일치"

    # 3. 병원명 포함 관계
    lengths = {
        pos: abs(len(names[pos]) - len(master_norm))
        for pos in positions
        if names[pos] and (master_norm in names[pos] or names[pos] in master_norm)
    }
    if lengths:
        # Keep the same pandas sort, including its tie ordering.
        pos = pd.Series(lengths).sort_values().index[0]
        api_row = realtime_df.iloc[pos].to_dict()
        return api_row, "name_contains", 88, "정규화 병원명 포함 관계 매칭"

    # 4. fuzzy matching
    best_position = None
    best_score = 0.0
    master_addr = str(master_row.get("_master_norm_address", "")).strip()

    normalized_addresses = search_columns["_api_norm_address"]
    for position in positions:
        api_norm = names[position].strip()

        if not api_norm:
            continue

        score = difflib.SequenceMatcher(None, master_norm, api_norm).ratio()

        # 주소 유사성이 있으면 약간 가산
        api_addr = normalized_addresses[position].strip()

        if master_addr and api_addr:
            addr_score = difflib.SequenceMatcher(None, master_addr[:12], api_addr[:12]).ratio()
            score = (score * 0.8) + (addr_score * 0.2)

        if score > best_score:
            best_score = score
            best_position = position

    if best_position is not None and best_score >= FUZZY_THRESHOLD:
        best_row = realtime_df.iloc[best_position].to_dict()
        return (
            best_row,
            "fuzzy_name_region",
            round(best_score * 100, 1),
            f"병원명/지역 유사도 매칭: {master_name} ↔ {best_row.get('_api_name', '')}",
        )

    return None, "unmatched", round(best_score * 100, 1), "실시간 응급의료기관 API에서 자동 매칭 실패"


def merge_master_with_realtime(master_df, realtime_df):
    """
    전국 병원 마스터 DB와 실시간 API 결과를 병합한다.
    """
    print("[3/4] 전국 병원 마스터 DB와 실시간 API 매칭 중...")

    master = prepare_master_df(master_df)
    # Keep row order and use positional IDs, including when source labels repeat.
    realtime = prepare_realtime_df(realtime_df).reset_index(drop=True)
    search_columns = {
        col: realtime[col].tolist()
        for col in ("_api_address", "_api_norm_address", "_api_norm_name")
    }
    search_columns["api_query_sido"] = (
        realtime["api_query_sido"].astype(str).apply(normalize_sido).tolist()
        if "api_query_sido" in realtime.columns else [""] * len(realtime)
    )
    all_positions = range(len(realtime))

    merged_rows = []

    for idx, master_row_series in master.iterrows():
        master_row = master_row_series.to_dict()
        hospital_name = master_row.get("_master_name", "")

        if (idx + 1) % 50 == 0:
            print(f"  - {idx + 1}/{len(master)}개 병원 처리 중...")

        api_row, method, score, note = find_best_match(master_row, realtime, search_columns, all_positions)

        if api_row is None:
            merged = mark_unmatched(master_row, note)
        else:
            merged = copy_realtime_values(
                master_row,
                api_row,
                match_status="OK",
                match_method=method,
                match_score=score,
                match_note=note,
            )

        merged_rows.append(merged)

    merged_df = pd.DataFrame(merged_rows)

    # 내부 보조 컬럼 제거
    drop_cols = [
        "_master_name",
        "_master_address",
        "_master_norm_name",
        "_master_norm_address",
        "_master_hpid",
        "_master_sido_short",
        "_api_name",
        "_api_norm_name",
        "_api_address",
        "_api_norm_address",
        "_api_hpid",
    ]

    merged_df = merged_df.drop(columns=[c for c in drop_cols if c in merged_df.columns], errors="ignore")

    # 컬럼 순서 정리
    preferred_cols = [
        "hospital_name",
        "target_hospital",
        "hospital_type",
        "sido",
        "sigungu",
        "address",
        "main_tel",
        "latitude",
        "longitude",
        "encrypted_ykiho",
        "hpid",
        "realtime_match_status",
        "match_method",
        "match_score",
        "match_note",
        "api_matched_name",
        "hvidate",
        "hvec",
        "hvoc",
        "hvgc",
        "hvicc",
        "hvncc",
        "hvcc",
        "hvccc",
        "hv2",
        "hv3",
        "hv4",
        "hv5",
        "hv6",
        "hvctayn",
        "hvmriayn",
        "hvangioayn",
        "hvventiayn",
        "data_fetched_at",
        "departments",
        "departments_source_url",
        "departments_last_checked_at",
        "departments_confidence",
        "departments_update_method",
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
        "source_api",
        "master_last_updated_at",
    ]

    ordered_cols = [c for c in preferred_cols if c in merged_df.columns]
    other_cols = [c for c in merged_df.columns if c not in ordered_cols]

    merged_df = merged_df[ordered_cols + other_cols]

    matched_count = int((merged_df["realtime_match_status"].astype(str) == "OK").sum())
    unmatched_count = len(merged_df) - matched_count

    print(f"    - 매칭 성공: {matched_count}개")
    print(f"    - 미매칭: {unmatched_count}개")

    return merged_df


# =========================================================
# 5. 파일 읽기/저장
# =========================================================

def choose_master_sheet(xls):

    preferred_sheets = [
        "hospital_master",
        "national_hospital_master",
        "national_hospital_db",
        "hospital_master_realtime",
        "hospital_db_realtime",
    ]

    for sheet in preferred_sheets:
        if sheet in xls.sheet_names:
            return sheet

    return xls.sheet_names[0]


def load_master_db():
    print("[1/4] 전국 병원 마스터 DB 읽는 중...")

    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} 파일이 없습니다. 먼저 build_national_master_db.py를 실행하세요."
        )

    with pd.ExcelFile(INPUT_FILE) as workbook:
        sheet_name = choose_master_sheet(workbook)
        master = workbook.parse(sheet_name)

    print(f"    - 파일: {INPUT_FILE}")
    print(f"    - 시트: {sheet_name}")
    print(f"    - 병원 수: {len(master)}개")

    return master


def save_output(merged_df, realtime_raw):
    print("[4/4] 결과 저장 중...")

    matched = merged_df[merged_df["realtime_match_status"].astype(str) == "OK"].copy()
    unmatched = merged_df[merged_df["realtime_match_status"].astype(str) != "OK"].copy()

    summary = pd.DataFrame(
        [
            {
                "total_hospitals": len(merged_df),
                "matched_hospitals": len(matched),
                "unmatched_hospitals": len(unmatched),
                "er_available_hospitals": int((pd.to_numeric(merged_df.get("hvec"), errors="coerce") > 0).sum())
                if "hvec" in merged_df.columns else 0,
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "note": "전국 상급종합병원/종합병원 마스터 DB와 응급의료기관 실시간 API 매칭 결과",
            }
        ]
    )

    match_method_summary = pd.DataFrame()

    if "match_method" in merged_df.columns:
        match_method_summary = (
            merged_df["match_method"]
            .fillna("unknown")
            .value_counts()
            .reset_index()
        )
        match_method_summary.columns = ["match_method", "count"]

    if find_spec("xlsxwriter") is None:
        engine = "openpyxl"
        engine_kwargs = {}
        print("    - 저장 엔진: openpyxl (xlsxwriter 미설치)")
    else:
        engine = "xlsxwriter"
        engine_kwargs = {"options": {"strings_to_urls": False}}
        print("    - 저장 엔진: xlsxwriter")

    with pd.ExcelWriter(OUTPUT_FILE, engine=engine, engine_kwargs=engine_kwargs) as writer:
        merged_df.to_excel(writer, index=False, sheet_name="national_hospital_db")
        matched.to_excel(writer, index=False, sheet_name="matched_only")
        unmatched.to_excel(writer, index=False, sheet_name="unmatched_only")
        summary.to_excel(writer, index=False, sheet_name="summary")
        match_method_summary.to_excel(writer, index=False, sheet_name="match_method_summary")
        realtime_raw.to_excel(writer, index=False, sheet_name="realtime_api_raw")

    print(f"저장 완료: {OUTPUT_FILE}")

    print()
    print("요약")
    print(f"- 전체 병원 수: {len(merged_df)}개")
    print(f"- 매칭 성공: {len(matched)}개")
    print(f"- 미매칭/정보없음: {len(unmatched)}개")

    if "hvec" in merged_df.columns:
        hvec_num = pd.to_numeric(merged_df["hvec"], errors="coerce")
        print(f"- 응급실 가용병상 있음: {int((hvec_num > 0).sum())}개")

    if not match_method_summary.empty:
        print()
        print("매칭 방식별 건수")
        for _, row in match_method_summary.iterrows():
            print(f"- {row['match_method']}: {row['count']}개")


# =========================================================
# 6. 메인 실행
# =========================================================

def main():
    print("==============================================")
    print("전국 병원 DB 실시간 응급자원 업데이트 시작")
    print("==============================================")

    started_at = time.perf_counter()
    master = load_master_db()
    master_loaded_at = time.perf_counter()

    realtime_raw = fetch_national_realtime_api()
    api_finished_at = time.perf_counter()

    if realtime_raw.empty:
        raise RuntimeError("실시간 API 응답이 비어 있습니다. 이전 파일을 이번 갱신 결과로 반영하지 않습니다.")

    merged_df = merge_master_with_realtime(master, realtime_raw)
    matching_finished_at = time.perf_counter()

    save_output(merged_df, realtime_raw)
    finished_at = time.perf_counter()

    print(
        f"[시간 측정] 마스터 읽기 {master_loaded_at - started_at:.2f}초 / "
        f"전국 API 수집 {api_finished_at - master_loaded_at:.2f}초 / "
        f"병원 매칭 {matching_finished_at - api_finished_at:.2f}초 / "
        f"엑셀 저장 {finished_at - matching_finished_at:.2f}초 / "
        f"스크립트 합계 {finished_at - started_at:.2f}초"
    )

    print("==============================================")
    print("작업 완료")
    print("==============================================")


if __name__ == "__main__":
    main()

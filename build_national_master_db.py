import os
import time
import math
import requests
import pandas as pd
import xml.etree.ElementTree as ET

from datetime import datetime
from urllib.parse import urlencode
from dotenv import load_dotenv


# =========================================================
# 1. 기본 설정
# =========================================================

load_dotenv()

HIRA_SERVICE_KEY = os.getenv("HIRA_SERVICE_KEY")

if not HIRA_SERVICE_KEY:
    raise RuntimeError(
        ".env 파일에 HIRA_SERVICE_KEY가 없습니다.\n"
        "예: HIRA_SERVICE_KEY=건강보험심사평가원_병원정보서비스_API_인증키"
    )

# 건강보험심사평가원 병원정보서비스
# 병원 기본 목록 조회용
BASE_URL = "https://apis.data.go.kr/B551182/hospInfoServicev2/getHospBasisList"

OUTPUT_FILE = "national_hospital_master.xlsx"

# 한 번에 너무 많이 요청하면 timeout이 날 수 있어서 300개씩 가져옴
NUM_OF_ROWS = 300

# 전국 시도 코드/이름
SIDO_LIST = [
    "서울",
    "부산",
    "대구",
    "인천",
    "광주",
    "대전",
    "울산",
    "세종",
    "경기",
    "강원",
    "충북",
    "충남",
    "전북",
    "전남",
    "경북",
    "경남",
    "제주",
]


# =========================================================
# 2. XML 파싱 유틸
# =========================================================

def get_text(elem, tag):
    """
    XML element에서 특정 tag의 텍스트를 안전하게 가져오는 함수.
    """
    found = elem.find(tag)

    if found is None:
        return ""

    if found.text is None:
        return ""

    return found.text.strip()


def parse_hira_xml(xml_text):
    """
    HIRA 병원정보 XML 응답을 DataFrame으로 변환한다.

    반환:
    - result_code
    - result_msg
    - total_count
    - df
    """
    root = ET.fromstring(xml_text)

    header = root.find(".//header")
    body = root.find(".//body")

    result_code = get_text(header, "resultCode") if header is not None else ""
    result_msg = get_text(header, "resultMsg") if header is not None else ""

    total_count_text = get_text(body, "totalCount") if body is not None else "0"

    try:
        total_count = int(total_count_text)
    except ValueError:
        total_count = 0

    rows = []

    for item in root.findall(".//item"):
        row = {}

        for child in list(item):
            row[child.tag] = child.text.strip() if child.text else ""

        rows.append(row)

    df = pd.DataFrame(rows)

    return result_code, result_msg, total_count, df


# =========================================================
# 3. API 호출
# =========================================================

def call_hira_api(page_no=1, num_of_rows=NUM_OF_ROWS, sido_name=None):
    """
    HIRA 병원정보서비스 API 호출.

    timeout이 자주 발생할 수 있어서:
    - timeout 90초
    - 최대 3회 재시도
    - 재시도 사이 5초 대기
    """

    params = {
        "pageNo": str(page_no),
        "numOfRows": str(num_of_rows),
    }

    # 시도별로 나눠 조회하고 싶을 때 사용
    # HIRA 병원정보서비스의 시도 파라미터는 sidoCd가 정석이지만,
    # 여기서는 기본 전국 조회를 우선 사용한다.
    # 필요하면 추후 sidoCd 매핑으로 변경 가능.
    if sido_name:
        # 일부 API는 sidoCd를 요구하므로, sido_name은 현재 로그/확장용으로만 둠.
        pass

    query = urlencode(params)

    # 인증키가 이미 인코딩된 형태일 수 있으므로 serviceKey는 직접 붙임
    full_url = f"{BASE_URL}?serviceKey={HIRA_SERVICE_KEY}&{query}"

    last_error = None

    for attempt in range(1, 4):
        try:
            print(
                f"    - API 호출 시도 {attempt}/3 "
                f"(pageNo={page_no}, numOfRows={num_of_rows})"
            )

            response = requests.get(full_url, timeout=90)
            response.raise_for_status()

            return response.text

        except requests.exceptions.ReadTimeout as e:
            last_error = e
            print(
                f"    ! ReadTimeout 발생: API 응답이 늦습니다. "
                f"5초 후 재시도합니다."
            )
            time.sleep(5)

        except requests.exceptions.ConnectionError as e:
            last_error = e
            print(
                f"    ! ConnectionError 발생: 네트워크 연결 문제가 있습니다. "
                f"5초 후 재시도합니다."
            )
            time.sleep(5)

        except requests.exceptions.RequestException as e:
            last_error = e
            print(
                f"    ! API 호출 오류: {e} "
                f"5초 후 재시도합니다."
            )
            time.sleep(5)

    raise RuntimeError(f"HIRA API 호출이 3회 실패했습니다: {last_error}")


# =========================================================
# 4. 전체 병원 목록 수집
# =========================================================

def fetch_all_hospitals():
    """
    전국 병원 기본 목록을 전체 페이지 수집한다.
    """

    print("[1/4] HIRA 병원기본목록 1페이지 호출 중...")

    first_xml = call_hira_api(page_no=1, num_of_rows=NUM_OF_ROWS)
    code, msg, total_count, first_df = parse_hira_xml(first_xml)

    print(f"    - resultCode: {code}")
    print(f"    - resultMsg: {msg}")
    print(f"    - totalCount: {total_count}")
    print(f"    - 1페이지 수집 건수: {len(first_df)}")

    if code not in ["00", "0000", "NORMAL SERVICE."]:
        print("    ! API 응답 코드가 정상 코드가 아닐 수 있습니다. 응답 내용을 확인하세요.")

    if total_count == 0:
        print("    ! totalCount가 0입니다. API 키, 활용신청 승인 여부, URL을 확인하세요.")
        return first_df

    total_pages = math.ceil(total_count / NUM_OF_ROWS)

    print(f"[2/4] 전체 {total_pages}페이지 수집 시작...")

    all_dfs = [first_df]

    for page_no in range(2, total_pages + 1):
        print(f"  - {page_no}/{total_pages} 페이지 호출 중...")

        try:
            xml_text = call_hira_api(page_no=page_no, num_of_rows=NUM_OF_ROWS)
            _, _, _, df = parse_hira_xml(xml_text)

            print(f"    수집 건수: {len(df)}")
            all_dfs.append(df)

            # 공공 API 서버 부담 줄이기
            time.sleep(0.3)

        except Exception as e:
            print(f"    ! {page_no}페이지 수집 실패: {e}")
            print("    ! 해당 페이지는 건너뛰고 계속 진행합니다.")
            continue

    if not all_dfs:
        return pd.DataFrame()

    raw_df = pd.concat(all_dfs, ignore_index=True)

    # 완전 중복 제거
    raw_df = raw_df.drop_duplicates()

    print(f"    - 전체 수집 완료: {len(raw_df)}건")

    return raw_df


# =========================================================
# 5. 상급종합병원 / 종합병원 필터링
# =========================================================

def filter_general_hospitals(raw_df):
    """
    전국 병원 목록에서 상급종합병원, 종합병원만 남긴다.
    HIRA 응답 컬럼명은 보통 clCdNm 또는 clCd가 사용된다.
    """

    if raw_df.empty:
        return raw_df

    df = raw_df.copy()

    print("[3/4] 상급종합병원/종합병원 필터링 중...")

    if "clCdNm" not in df.columns:
        print("    ! clCdNm 컬럼이 없습니다. 실제 컬럼명을 확인하세요.")
        print(f"    - 현재 컬럼: {list(df.columns)}")
        return df

    type_col = df["clCdNm"].astype(str)

    mask = (
        type_col.str.contains("상급종합", na=False)
        | type_col.str.contains("종합병원", na=False)
    )

    filtered = df[mask].copy()

    print(f"    - 필터링 전: {len(df)}건")
    print(f"    - 필터링 후: {len(filtered)}건")

    return filtered


# =========================================================
# 6. 컬럼 정리
# =========================================================

def pick_first_existing(row, candidates):
    """
    후보 컬럼 중 값이 존재하는 첫 번째 값을 반환한다.
    """
    for col in candidates:
        if col in row and str(row.get(col)).strip() != "":
            return row.get(col)

    return ""


def build_master_columns(filtered_df):
    """
    기존 프로젝트에서 쓰기 좋은 전국 병원 마스터 DB 컬럼으로 정리한다.
    """

    print("[4/4] 마스터 DB 컬럼 정리 중...")

    rows = []

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for _, row in filtered_df.iterrows():
        row_dict = row.to_dict()

        hospital_name = pick_first_existing(row_dict, ["yadmNm", "dutyName", "dutyname"])
        hospital_type = pick_first_existing(row_dict, ["clCdNm"])
        address = pick_first_existing(row_dict, ["addr", "dutyAddr", "dutyaddr"])
        main_tel = pick_first_existing(row_dict, ["telno", "dutyTel1", "dutyTel"])
        sido = pick_first_existing(row_dict, ["sidoCdNm"])
        sigungu = pick_first_existing(row_dict, ["sgguCdNm"])
        latitude = pick_first_existing(row_dict, ["YPos", "wgs84Lat", "latitude"])
        longitude = pick_first_existing(row_dict, ["XPos", "wgs84Lon", "longitude"])
        encrypted_ykiho = pick_first_existing(row_dict, ["ykiho", "encryptedYkiho"])
        hpid = pick_first_existing(row_dict, ["hpid", "HPID"])

        rows.append(
            {
                # 기본 식별 정보
                "hospital_name": hospital_name,
                "target_hospital": hospital_name,
                "hospital_type": hospital_type,
                "sido": sido,
                "sigungu": sigungu,
                "address": address,
                "main_tel": main_tel,
                "latitude": latitude,
                "longitude": longitude,
                "encrypted_ykiho": encrypted_ykiho,
                "hpid": hpid,

                # 진료과 AI 서칭용 컬럼
                "departments": "",
                "departments_source_url": "",
                "departments_last_checked_at": "",
                "departments_confidence": "",
                "departments_update_method": "",

                # 주요 진료과 boolean 컬럼
                "has_emergency_medicine": "",
                "has_neurology": "",
                "has_neurosurgery": "",
                "has_cardiology": "",
                "has_cardiovascular_medicine": "",
                "has_general_surgery": "",
                "has_thoracic_surgery": "",
                "has_orthopedics": "",
                "has_obgyn": "",
                "has_pediatrics": "",
                "has_neonatology": "",

                # 응급의료 실시간 API 매칭용 컬럼
                "realtime_match_status": "",
                "api_matched_name": "",

                # 실시간 응급 자원 컬럼
                "hvidate": "",
                "hvec": "",
                "hvoc": "",
                "hvgc": "",
                "hvicc": "",
                "hvncc": "",
                "hvcc": "",
                "hvccc": "",
                "hv2": "",
                "hv3": "",
                "hv4": "",
                "hv5": "",
                "hv6": "",
                "hvctayn": "",
                "hvmriayn": "",
                "hvangioayn": "",
                "hvventiayn": "",
                "data_fetched_at": "",

                # 출처/관리
                "source_api": "HIRA_hospInfoServicev2_getHospBasisList",
                "master_last_updated_at": now,
            }
        )

    master_df = pd.DataFrame(rows)

    # 병원명 없는 행 제거
    master_df = master_df[master_df["hospital_name"].astype(str).str.strip() != ""].copy()

    # 병원명 + 주소 기준 중복 제거
    master_df = master_df.drop_duplicates(subset=["hospital_name", "address"])

    # 정렬
    sort_cols = [col for col in ["sido", "sigungu", "hospital_type", "hospital_name"] if col in master_df.columns]
    master_df = master_df.sort_values(by=sort_cols).reset_index(drop=True)

    return master_df


# =========================================================
# 7. 저장
# =========================================================

def save_master_excel(master_df, raw_df, filtered_df):
    """
    national_hospital_master.xlsx 저장.
    """

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        master_df.to_excel(
            writer,
            index=False,
            sheet_name="hospital_master",
        )

        filtered_df.to_excel(
            writer,
            index=False,
            sheet_name="filtered_raw_hira",
        )

        raw_df.to_excel(
            writer,
            index=False,
            sheet_name="raw_hira_all",
        )

    print(f"저장 완료: {OUTPUT_FILE}")
    print(f"전국 상급종합병원/종합병원 수: {len(master_df)}")


# =========================================================
# 8. 메인 실행
# =========================================================

def main():
    print("==============================================")
    print("전국 상급종합병원/종합병원 마스터 DB 생성 시작")
    print("==============================================")

    raw_df = fetch_all_hospitals()

    if raw_df.empty:
        print("수집된 병원 데이터가 없습니다. 작업을 종료합니다.")
        return

    filtered_df = filter_general_hospitals(raw_df)

    if filtered_df.empty:
        print("상급종합병원/종합병원 필터링 결과가 없습니다.")
        print("raw_hira_all 데이터를 확인해야 합니다.")
        return

    master_df = build_master_columns(filtered_df)

    save_master_excel(master_df, raw_df, filtered_df)

    print("==============================================")
    print("작업 완료")
    print("다음 단계:")
    print("python3 update_realtime_resources_national.py")
    print("==============================================")


if __name__ == "__main__":
    main()
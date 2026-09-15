import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st


# =========================================================
# 1. 기본 설정
# =========================================================

st.set_page_config(
    page_title="전국 응급의료 병원 DB 실시간 업데이트",
    layout="wide",
)

st.title("응급의료 병원 DB 실시간 업데이트 프로토타입")
st.caption(
    "전국 상급종합병원/종합병원 DB를 기준으로 공공데이터 API를 호출해 "
    "병원별 실시간 가용 정보와 진료과/전문의/특수진료/의료장비 정보를 관리하는 구조입니다."
)

DEFAULT_FILE = "national_hospital_db_departments_hira_updated.xlsx"
REALTIME_FILE = "national_hospital_db_realtime_updated.xlsx"
MASTER_FILE = "national_hospital_master.xlsx"
SAVER_DB_FILE = "saver_current_hospital_db.xlsx"

NATIONAL_UPDATE_SCRIPT = "update_realtime_resources_national.py"
DEPARTMENT_UPDATE_SCRIPT = "update_departments_hira_api.py"

RECOMMENDATION_SYSTEM_URL = os.getenv("RECOMMENDATION_SYSTEM_URL", "http://127.0.0.1:5050")

# 화면/저장 파일에서 숨길 컬럼
# 원본 API 매칭 검증용 컬럼이지만, 시연 화면에서는 수용 점수와 혼동될 수 있어서 제외
HIDDEN_COLUMNS = [
    "match_score",
    "match_method",
    "match_note",
]


# =========================================================
# 2. 세션 상태 초기화
# =========================================================

default_states = {
    "hospital_df": None,
    "last_fetch": None,
    "fetch_error": None,
    "run_status": "대기 중",
    "run_log": [],
    "script_stdout": "",
    "script_stderr": "",
    "excel_saved": False,
}

for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 3. 공통 함수
# =========================================================

def add_log(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.run_log.append(f"[{now}] {message}")
    st.session_state.run_log = st.session_state.run_log[-40:]


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


def remove_hidden_columns(df):
    df = df.drop(
        columns=[col for col in df.columns if str(col).startswith("Unnamed")],
        errors="ignore",
    )

    df = df.drop(
        columns=[col for col in HIDDEN_COLUMNS if col in df.columns],
        errors="ignore",
    )

    return df


def format_hvidate(df):
    if "hvidate" in df.columns:
        df["hvidate"] = df["hvidate"].apply(
            lambda x: "" if pd.isna(x) else str(x).split(".")[0]
        )

    return df


def load_hospital_db(excel_file):
    if not os.path.exists(excel_file):
        raise FileNotFoundError(f"{excel_file} 파일이 없습니다.")

    sheet_name = choose_sheet_name(excel_file)
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    df = remove_hidden_columns(df)
    df = format_hvidate(df)

    return df, sheet_name


def reset_execution_state():
    st.session_state.fetch_error = None
    st.session_state.run_status = "수행 정지"
    st.session_state.script_stdout = ""
    st.session_state.script_stderr = ""
    st.session_state.excel_saved = False
    add_log("수행 멈춤: 화면 실행 상태 초기화")


def run_script(script_name, env=None):
    script_path = Path(script_name)

    if not script_path.exists():
        raise FileNotFoundError(f"{script_name} 파일이 없습니다.")

    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True,
        cwd=".",
        env=env or os.environ.copy(),
    )

    st.session_state.script_stdout = result.stdout
    st.session_state.script_stderr = result.stderr

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} 실행 중 오류가 발생했습니다.\n\n"
            f"[STDOUT]\n{result.stdout}\n\n"
            f"[STDERR]\n{result.stderr}"
        )


def refresh_realtime_data():
    st.session_state.run_status = "전국 실시간 API 갱신 중"
    st.session_state.fetch_error = None
    st.session_state.excel_saved = False
    add_log("전국 실시간 응급자원 API 데이터 새로고침 시작")

    try:
        if not Path(MASTER_FILE).exists():
            raise FileNotFoundError(
                f"{MASTER_FILE} 파일이 없습니다. 먼저 build_national_master_db.py를 실행하세요."
            )

        run_script(NATIONAL_UPDATE_SCRIPT)

        if not Path(REALTIME_FILE).exists():
            raise FileNotFoundError(
                f"{NATIONAL_UPDATE_SCRIPT} 실행 후에도 {REALTIME_FILE} 파일이 생성되지 않았습니다."
            )

        df, sheet_name = load_hospital_db(REALTIME_FILE)

        st.session_state.hospital_df = df
        st.session_state.last_fetch = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.run_status = "실시간 갱신 완료"
        st.session_state.excel_saved = False

        add_log(f"실시간 API 갱신 완료: {len(df)}개 병원 / 시트 {sheet_name}")
        add_log("SAVER 반영을 위해 [엑셀로 저장]을 눌러주세요.")

    except Exception as e:
        st.session_state.fetch_error = str(e)
        st.session_state.run_status = "오류 발생"
        add_log(f"실시간 API 갱신 오류: {e}")


def refresh_department_data():
    st.session_state.run_status = "HIRA 상세정보 갱신 중"
    st.session_state.fetch_error = None
    st.session_state.excel_saved = False
    add_log("HIRA 진료과/전문의/특수진료/의료장비 갱신 시작")

    try:
        if not Path(DEPARTMENT_UPDATE_SCRIPT).exists():
            raise FileNotFoundError(f"{DEPARTMENT_UPDATE_SCRIPT} 파일이 없습니다.")

        env = os.environ.copy()
        env["DEPARTMENT_UPDATE_LIMIT"] = env.get("DEPARTMENT_UPDATE_LIMIT", "0")

        run_script(DEPARTMENT_UPDATE_SCRIPT, env=env)

        if not Path(DEFAULT_FILE).exists():
            raise FileNotFoundError(
                f"{DEPARTMENT_UPDATE_SCRIPT} 실행 후에도 {DEFAULT_FILE} 파일이 생성되지 않았습니다."
            )

        df, sheet_name = load_hospital_db(DEFAULT_FILE)

        st.session_state.hospital_df = df
        st.session_state.last_fetch = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.run_status = "HIRA 상세정보 갱신 완료"
        st.session_state.excel_saved = False

        add_log(f"HIRA 상세정보 갱신 완료: {len(df)}개 병원 / 시트 {sheet_name}")
        add_log("SAVER 반영을 위해 [엑셀로 저장]을 눌러주세요.")

    except Exception as e:
        st.session_state.fetch_error = str(e)
        st.session_state.run_status = "오류 발생"
        add_log(f"HIRA 상세정보 갱신 오류: {e}")


def load_current_file_to_session(db_file):
    try:
        df, sheet_name = load_hospital_db(db_file)
        st.session_state.hospital_df = df
        st.session_state.fetch_error = None
        st.session_state.excel_saved = False
        add_log(f"파일 로드 완료: {db_file} / 시트 {sheet_name} / {len(df)}개 병원")
        add_log("SAVER 반영을 위해 [엑셀로 저장]을 눌러주세요.")

    except Exception as e:
        st.session_state.fetch_error = str(e)
        st.session_state.run_status = "오류 발생"
        add_log(f"파일 로드 오류: {e}")


def save_current_db_for_app_and_saver():
    if st.session_state.hospital_df is None:
        st.warning("저장할 데이터가 없습니다.")
        add_log("엑셀 저장 실패: 데이터 없음")
        return

    output_name = DEFAULT_FILE
    saver_output_name = SAVER_DB_FILE

    df_to_save = st.session_state.hospital_df.copy()
    df_to_save = remove_hidden_columns(df_to_save)
    df_to_save = format_hvidate(df_to_save)

    with pd.ExcelWriter(output_name, engine="openpyxl") as writer:
        df_to_save.to_excel(
            writer,
            index=False,
            sheet_name="national_hospital_db",
        )

        meta = pd.DataFrame(
            [
                {
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "Streamlit app.py",
                    "note": "전국 상급종합병원/종합병원 최종 DB 저장",
                    "hidden_columns": ", ".join(HIDDEN_COLUMNS),
                }
            ]
        )
        meta.to_excel(writer, index=False, sheet_name="save_meta")

    with pd.ExcelWriter(saver_output_name, engine="openpyxl") as writer:
        df_to_save.to_excel(
            writer,
            index=False,
            sheet_name="national_hospital_db",
        )

        meta = pd.DataFrame(
            [
                {
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "Streamlit app.py",
                    "note": "SAVER 추천 시스템이 읽는 최신 병원 DB",
                    "base_file": output_name,
                    "hidden_columns": ", ".join(HIDDEN_COLUMNS),
                }
            ]
        )
        meta.to_excel(writer, index=False, sheet_name="saver_meta")

    st.session_state.excel_saved = True
    st.session_state.run_status = "엑셀 저장 완료"
    st.session_state.last_fetch = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    add_log(f"최종 DB 저장 완료: {output_name}")
    add_log(f"SAVER 최신 DB 저장 완료: {saver_output_name}")

    st.toast("최신 DB 저장 완료. SAVER에서 바로 사용할 수 있습니다.", icon="✅")


def build_summary(df):
    total_count = len(df)

    matched_count = 0
    unmatched_count = 0

    if "realtime_match_status" in df.columns:
        matched_count = int((df["realtime_match_status"].astype(str) == "OK").sum())
        unmatched_count = total_count - matched_count
    elif "match_status" in df.columns:
        matched_count = int((df["match_status"].astype(str) == "OK").sum())
        unmatched_count = total_count - matched_count

    er_available_count = 0
    if "hvec" in df.columns:
        hvec_num = pd.to_numeric(df["hvec"], errors="coerce")
        er_available_count = int((hvec_num > 0).sum())

    departments_count = 0
    if "departments" in df.columns:
        departments_count = int(df["departments"].astype(str).str.strip().ne("").sum())

    specialist_count_columns = [
        col for col in df.columns
        if str(col).startswith("specialist_count_")
    ]

    hospitals_with_specialist_count = 0
    if specialist_count_columns:
        specialist_num_df = df[specialist_count_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        hospitals_with_specialist_count = int(
            specialist_num_df.notna().any(axis=1).sum()
        )

    special_diag_count = 0
    if "special_diag_summary" in df.columns:
        special_diag_count = int(
            df["special_diag_summary"].astype(str).str.strip().ne("").sum()
        )

    equipment_count = 0
    if "medical_equipment_summary" in df.columns:
        equipment_count = int(
            df["medical_equipment_summary"].astype(str).str.strip().ne("").sum()
        )

    return (
        total_count,
        matched_count,
        unmatched_count,
        er_available_count,
        departments_count,
        hospitals_with_specialist_count,
        special_diag_count,
        equipment_count,
    )


# =========================================================
# 4. 사이드바
# =========================================================

with st.sidebar:
    st.header("설정")

    db_file = st.text_input(
        "병원 DB 파일명",
        value=DEFAULT_FILE,
    )

    st.caption("기본값은 HIRA 상세정보까지 보완된 최종 DB입니다.")

    st.divider()

    st.subheader("현재 수행 상태")

    if st.session_state.run_status in ["실시간 갱신 완료", "HIRA 상세정보 갱신 완료", "엑셀 저장 완료"]:
        st.success(st.session_state.run_status)
    elif "갱신 중" in st.session_state.run_status:
        st.warning(st.session_state.run_status)
    elif st.session_state.run_status == "오류 발생":
        st.error(st.session_state.run_status)
    elif st.session_state.run_status == "수행 정지":
        st.info(st.session_state.run_status)
    else:
        st.info(st.session_state.run_status)

    st.write(
        "서버 종료는 터미널에서 Ctrl + C로 수행합니다. "
        "앱 화면의 수행 멈춤 버튼은 화면 실행 상태를 초기화하는 기능입니다."
    )

    st.divider()

    if st.button("현재 파일 다시 불러오기", use_container_width=True):
        load_current_file_to_session(db_file)


# =========================================================
# 5. 초기 파일 로드
# =========================================================

if st.session_state.hospital_df is None:
    if Path(db_file).exists():
        load_current_file_to_session(db_file)
    elif Path(DEFAULT_FILE).exists():
        load_current_file_to_session(DEFAULT_FILE)
    elif Path(REALTIME_FILE).exists():
        load_current_file_to_session(REALTIME_FILE)
    elif Path(MASTER_FILE).exists():
        load_current_file_to_session(MASTER_FILE)
    else:
        st.error(
            "전국 병원 DB 파일이 없습니다. 먼저 아래 명령어를 실행하세요.\n\n"
            "python3 build_national_master_db.py\n"
            "python3 update_realtime_resources_national.py\n"
            "DEPARTMENT_UPDATE_LIMIT=0 python3 update_departments_hira_api.py"
        )
        st.stop()


# =========================================================
# 6. 상단 버튼
# =========================================================

col1, col2, col3, col4, col5 = st.columns([1.2, 1.2, 1.2, 1, 2.4])

with col1:
    if st.button("실시간 API 새로고침", type="primary", use_container_width=True):
        refresh_realtime_data()

with col2:
    if st.button("HIRA 상세정보 갱신", use_container_width=True):
        refresh_department_data()

with col3:
    if st.button("엑셀로 저장", use_container_width=True):
        try:
            save_current_db_for_app_and_saver()
        except Exception as e:
            st.session_state.fetch_error = str(e)
            st.session_state.run_status = "오류 발생"
            add_log(f"엑셀 저장 오류: {e}")

    if st.session_state.excel_saved and Path(SAVER_DB_FILE).exists():
        st.link_button(
            "🚑 SAVER 실행",
            url=RECOMMENDATION_SYSTEM_URL,
            type="primary",
            use_container_width=True,
        )
        st.caption(f"저장 완료: {SAVER_DB_FILE}")

with col4:
    if st.button("수행 멈춤", use_container_width=True):
        reset_execution_state()

with col5:
    if st.session_state.last_fetch:
        st.info(f"마지막 갱신/로드/저장 시각: {st.session_state.last_fetch}")
    else:
        st.info("현재 파일을 기준으로 표시 중입니다.")


# =========================================================
# 7. 오류 표시
# =========================================================

if st.session_state.fetch_error:
    st.error("처리 중 오류가 났어요.")
    st.code(st.session_state.fetch_error)


# =========================================================
# 8. 데이터 준비
# =========================================================

hospital_df = st.session_state.hospital_df.copy()
hospital_df = remove_hidden_columns(hospital_df)
hospital_df = format_hvidate(hospital_df)
hospital_df = hospital_df.where(pd.notna(hospital_df), None)

(
    total_count,
    matched_count,
    unmatched_count,
    er_available_count,
    departments_count,
    hospitals_with_specialist_count,
    special_diag_count,
    equipment_count,
) = build_summary(hospital_df)


# =========================================================
# 9. 요약 카드
# =========================================================

st.subheader("전국 병원 DB 요약")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("전체 병원 수", f"{total_count:,}개")

with m2:
    st.metric("응급의료 API 매칭", f"{matched_count:,}개")

with m3:
    st.metric("미매칭/정보없음", f"{unmatched_count:,}개")

with m4:
    st.metric("응급실 가용병상 있음", f"{er_available_count:,}개")

m5, m6, m7, m8 = st.columns(4)

with m5:
    st.metric("진료과 정보 있음", f"{departments_count:,}개")

with m6:
    st.metric("전문의 수 정보 있음", f"{hospitals_with_specialist_count:,}개")

with m7:
    st.metric("특수진료정보 있음", f"{special_diag_count:,}개")

with m8:
    st.metric("의료장비정보 있음", f"{equipment_count:,}개")


# =========================================================
# 10. 필터
# =========================================================

st.subheader("병원별 실시간 가용 정보 및 HIRA 상세정보")

filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1, 1, 1, 2])

with filter_col1:
    selected_sido = st.selectbox(
        "시도 필터",
        options=["전체"] + sorted(
            [
                str(x)
                for x in hospital_df.get("sido", pd.Series(dtype=str)).dropna().unique()
                if str(x).strip() != ""
            ]
        ),
    )

with filter_col2:
    selected_match = st.selectbox(
        "API 매칭 상태",
        options=["전체", "OK", "미매칭/정보없음"],
    )

with filter_col3:
    selected_dept = st.selectbox(
        "진료과 정보",
        options=["전체", "있음", "없음"],
    )

with filter_col4:
    search_keyword = st.text_input(
        "병원명 검색",
        value="",
        placeholder="예: 강릉아산병원, 충남대학교병원",
    )


filtered_df = hospital_df.copy()

if selected_sido != "전체" and "sido" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["sido"].astype(str) == selected_sido]

if selected_match != "전체":
    status_col = None

    if "realtime_match_status" in filtered_df.columns:
        status_col = "realtime_match_status"
    elif "match_status" in filtered_df.columns:
        status_col = "match_status"

    if status_col:
        if selected_match == "OK":
            filtered_df = filtered_df[filtered_df[status_col].astype(str) == "OK"]
        else:
            filtered_df = filtered_df[filtered_df[status_col].astype(str) != "OK"]

if selected_dept != "전체" and "departments" in filtered_df.columns:
    has_dept = filtered_df["departments"].astype(str).str.strip().ne("")
    if selected_dept == "있음":
        filtered_df = filtered_df[has_dept]
    else:
        filtered_df = filtered_df[~has_dept]

if search_keyword.strip():
    keyword = search_keyword.strip()

    name_cols = [
        col
        for col in ["hospital_name", "target_hospital", "dutyname", "api_matched_name"]
        if col in filtered_df.columns
    ]

    if name_cols:
        mask = False
        for col in name_cols:
            mask = mask | filtered_df[col].astype(str).str.contains(keyword, na=False)
        filtered_df = filtered_df[mask]


# =========================================================
# 11. 표시 컬럼
# =========================================================

base_cols = [
    "hospital_name",
    "target_hospital",
    "hospital_type",
    "sido",
    "sigungu",
    "address",
    "main_tel",
    "hpid",
    "realtime_match_status",
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

specialist_count_cols = sorted(
    [
        col for col in filtered_df.columns
        if str(col).startswith("specialist_count_")
    ]
)

tail_cols = [
    "special_diag_summary",
    "medical_equipment_summary",
]

preferred_cols = base_cols + specialist_count_cols + tail_cols
display_cols = [col for col in preferred_cols if col in filtered_df.columns]

if not display_cols:
    display_cols = filtered_df.columns.tolist()

st.dataframe(
    filtered_df[display_cols],
    use_container_width=True,
    height=520,
)


# =========================================================
# 12. 매칭용 해석
# =========================================================

st.subheader("매칭용 해석")

st.write(
    "전국 상급종합병원/종합병원 마스터 DB에 응급의료기관 실시간 API 값을 병합하고, "
    "HIRA 의료기관별 상세정보서비스를 통해 진료과목, 전문과목별 전문의 수, 특수진료 가능 분야, 의료장비 정보를 보완합니다."
)

st.info(
    "realtime_match_status가 OK인 병원은 응급의료기관 실시간 API와 매칭된 병원입니다. "
    "미매칭 또는 빈값인 병원은 병원 기본 목록에는 존재하지만, 해당 API에서 실시간 응급자원 정보가 매칭되지 않은 병원으로 해석합니다."
)

st.warning(
    "진료과와 전문의 수는 실시간 수용 가능 여부가 아니라 병원의 기본 역량 정보입니다. "
    "SAVER 추천 단계에서는 필수 진료과 여부를 조건에 포함하되, 최종 이송 전에는 병원 사전 수용 확인이 필요합니다."
)


# =========================================================
# 13. API 실행 로그
# =========================================================

with st.expander("업데이트 스크립트 로그 보기"):
    if st.session_state.script_stdout:
        st.write("STDOUT")
        st.code(st.session_state.script_stdout)

    if st.session_state.script_stderr:
        st.write("STDERR")
        st.code(st.session_state.script_stderr)

    if not st.session_state.script_stdout and not st.session_state.script_stderr:
        st.write("아직 스크립트 실행 로그가 없습니다.")


# =========================================================
# 14. SAVER 추천 시스템 연결
# =========================================================

st.divider()

st.subheader("SAVER 추천 시스템 연결")

if st.session_state.excel_saved and Path(SAVER_DB_FILE).exists():
    st.success(f"최신 DB가 SAVER용 파일로 저장되었습니다: {SAVER_DB_FILE}")

    st.link_button(
        "🚑 SAVER 실행",
        url=RECOMMENDATION_SYSTEM_URL,
        type="primary",
        use_container_width=True,
    )

    st.caption(
        "이 버튼으로 이동하면 SAVER-AI는 saver_current_hospital_db.xlsx를 최우선으로 읽습니다."
    )

else:
    st.warning(
        "SAVER로 이동하기 전에 먼저 [엑셀로 저장]을 눌러 최신 병원 DB를 저장하세요."
    )

    st.link_button(
        "🚑 SAVER 화면만 열기",
        url=RECOMMENDATION_SYSTEM_URL,
        use_container_width=True,
    )


# =========================================================
# 15. 수행 로그
# =========================================================

st.subheader("수행 로그")

if st.session_state.run_log:
    for log in reversed(st.session_state.run_log):
        st.write(log)
else:
    st.write("아직 수행 로그가 없습니다.")
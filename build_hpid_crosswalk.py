"""Generate the reviewed-by-rule HPID crosswalk from the submitted workbooks."""
import pandas as pd
from update_realtime_resources_national import normalize_hospital_name, normalize_sido


def region(value):
    return "세종" if value == "세종시" else normalize_sido(value)


def main():
    realtime = "national_hospital_db_realtime_updated.xlsx"
    baseline = pd.read_excel("national_hospital_db_departments_hira_updated.xlsx", dtype=str)
    previous = pd.read_excel(realtime, dtype=str)
    raw = pd.read_excel(realtime, sheet_name="realtime_api_raw", dtype=str)
    previous = previous.loc[previous["match_method"].eq("name_exact"), ["encrypted_ykiho", "hpid"]]
    rows = baseline[["encrypted_ykiho", "hospital_name", "sido"]].merge(
        previous, on="encrypted_ykiho", validate="one_to_one"
    )
    api = raw[["hpid", "dutyName", "api_query_sido"]].drop_duplicates().copy()
    api["name"] = api["dutyName"].map(normalize_hospital_name)
    api["region"] = api["api_query_sido"].map(region)
    api = api[["hpid", "name", "region"]].drop_duplicates()
    api = api.loc[~api.duplicated(["name", "region"], keep=False)]
    rows["name"] = rows["hospital_name"].map(normalize_hospital_name)
    rows["region"] = rows["sido"].map(region)
    rows = rows.merge(api, on=["hpid", "name", "region"], validate="one_to_one")
    assert rows["hpid"].is_unique and rows["encrypted_ykiho"].is_unique
    rows[["encrypted_ykiho", "hpid", "hospital_name", "sido"]].to_csv(
        "data/nemc_hpid_crosswalk.csv", index=False
    )
    print(f"Unique exact-name and region matches: {len(rows)}/{len(baseline)}")


if __name__ == "__main__":
    main()

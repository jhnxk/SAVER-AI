import os
import tempfile
import threading
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd

import ai_server as server
import api_client
from realtime_store import KST, LiveHospitalStore


class LivePipelineTest(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 7, 2, 0, tzinfo=KST)
        with patch.dict(os.environ, {"IEUM_SNAPSHOT_DIR": "", "NEMC_MAX_AGE_SECONDS": "900"}):
            self.store = LiveHospitalStore(clock=lambda: self.now)
        self.store.initialize()
        self.rows = [
            {"hpid": hpid, "hvidate": "20261007015900", "hvec": "7",
             "hvoc": "2", "hvicc": "1", "hvctayn": "Y"}
            for hpid in sorted(self.store.known_ids)
        ]
        self.store.fetch = lambda: self.rows
        self.client = server.app.test_client()

    def test_snapshot_partial_failure_expiry_and_identity(self):
        self.assertEqual(len(self.rows), 338)
        self.assertTrue(self.store.refresh())
        df, meta = self.store.snapshot()
        self.assertEqual((meta["hospital_count"], meta["mapped_count"], meta["fresh_count"]), (384, 338, 338))
        self.assertTrue(df.loc[df["hpid"].eq("A2800020"), "hospital_name"].str.contains("삼성창원").all())
        wrong = df.loc[df["hospital_name"].str.contains("근로복지공단 창원병원")]
        self.assertEqual(wrong["hvec"].tolist(), [""])
        self.store.fetch = lambda: self.rows[:15]
        self.assertFalse(self.store.refresh())
        _, after = self.store.snapshot()
        self.assertEqual(after["status"], "degraded")
        self.assertEqual(meta["db_updated_at"], after["db_updated_at"])
        self.now += timedelta(minutes=16)
        df, meta = self.store.snapshot()
        self.assertEqual(meta["status"], "stale")
        self.assertTrue(df["hvec"].eq("").all())

    def test_delayed_refresh_serves_previous_snapshot(self):
        self.assertTrue(self.store.refresh())
        entered, release = threading.Event(), threading.Event()

        def slow_fetch():
            entered.set()
            release.wait(2)
            return [{**row, "hvidate": "20261007020000", "hvec": "9"} for row in self.rows]

        self.store.fetch = slow_fetch
        worker = threading.Thread(target=self.store.refresh)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertFalse(self.store.refresh())
            before, meta = self.store.snapshot()
            self.assertEqual(before.loc[before.hpid.ne(""), "hvec"].unique().tolist(), [7])
            self.assertEqual(meta["fresh_count"], 338)
        finally:
            release.set()
            worker.join(3)
        after, _ = self.store.snapshot()
        self.assertEqual(after.loc[after.hpid.ne(""), "hvec"].unique().tolist(), [9])

    def test_persisted_snapshot_remains_time_limited(self):
        with tempfile.TemporaryDirectory() as path:
            with patch.dict(os.environ, {"IEUM_SNAPSHOT_DIR": path}):
                first = LiveHospitalStore(fetch=lambda: self.rows, clock=lambda: self.now)
                first.initialize()
                self.assertTrue(first.refresh())
                restarted = LiveHospitalStore(clock=lambda: self.now)
                restarted.initialize()
                self.assertEqual(restarted.snapshot()[1]["fresh_count"], 338)
                self.now += timedelta(hours=1)
                self.assertEqual(restarted.snapshot()[1]["fresh_count"], 0)

    def test_route_metrics_and_provisional_order(self):
        rows = pd.DataFrame([
            {"eta_min": 5, "kakao_route_ok": False, "unknown_reasons_text": "", "model_score": 100},
            {"eta_min": 18, "kakao_route_ok": True, "unknown_reasons_text": "", "model_score": 80},
            {"eta_min": 8, "kakao_route_ok": True, "unknown_reasons_text": "장비 미확인", "model_score": 90},
        ])
        with patch.object(server, "LIVE_DB", True):
            ranked, _ = server.apply_golden_time_safety_priority(rows, {"golden_time_min": 10})
        self.assertEqual(ranked["eta_min"].tolist(), [18, 5, 8])
        self.assertEqual(ranked["golden_time_status"].tolist(), ["골든타임 초과", "ETA 미확인", "골든타임 내"])
        metrics = server.compute_performance_metrics(rows, rows, rows, {"golden_time_min": 10}, 1)
        self.assertEqual((metrics["confirmed_eta_count"], metrics["top_k_count"], metrics["golden_time_compliance_rate_pct"]), (2, 3, 50.0))

    def test_api_uses_fresh_snapshot_and_avoids_llm_reprocessing(self):
        patient = {"suspected_category": "가상 사례", "ktas": 3, "golden_time_min": 30,
                   "specialists": [], "equipment": [], "req_icu": False, "req_or": False,
                   "req_er_bed": True, "req_severe_acceptance": False}
        response = Mock(status_code=200)
        response.json.side_effect = lambda: {"routes": []}

        with patch.object(server, "LIVE_DB", True), patch.object(server, "LIVE_STORE", self.store), \
                patch.object(server, "analyze_patient", return_value=patient) as analyze, \
                patch.object(server, "KAKAO_REST_API_KEY", "test"), \
                patch.object(server.requests, "post", return_value=response):
            self.assertEqual(self.client.post("/api/recommend-hospitals", json={"text": "가상 발화"}).status_code, 503)
            analyze.assert_not_called()
            self.assertTrue(self.store.refresh())
            first = self.client.post("/api/recommend-hospitals", json={"text": "가상 발화"})
            self.assertEqual(first.status_code, 200)
            result = first.json
            self.assertEqual(result["db_status"]["fresh_count"], 338)
            self.assertGreater(result["display_hospital_count"], 0)
            self.assertEqual({h["realtime_match_status"] for h in result["acceptable_hospitals"]}, {"OK"})
            self.assertIsNone(result["performance_metrics"]["golden_time_compliance_rate_pct"])
            for model in "BCD":
                r = self.client.post("/api/recommend-hospitals", json={"patient": result["patient_input"], "model": model})
                self.assertEqual(r.status_code, 200)
                self.assertTrue(r.json["patient_analysis_reused"])
            analyze.assert_called_once()
            self.assertEqual(self.client.post("/api/sync-hospitals", json={"hospitals": [{}]}).status_code, 403)
            self.assertEqual(self.client.get("/./ai_server.py").status_code, 404)

    def test_nemc_error_does_not_disclose_key(self):
        with patch.dict(os.environ, {"SERVICE_KEY": "private%2Btest"}), \
                patch.object(api_client.requests, "get", side_effect=api_client.requests.Timeout("private+test")):
            with self.assertRaisesRegex(RuntimeError, r"^nemc_Timeout$"):
                api_client.fetch_national_beds()

    def test_nemc_complete_pages_only(self):
        def page(hpid):
            r = Mock(status_code=200)
            r.__enter__ = Mock(return_value=r)
            r.__exit__ = Mock(return_value=False)
            r.content = f"""<response><header><resultCode>00</resultCode></header>
                <body><items><item><hpid>{hpid}</hpid><hvidate>20261007015900</hvidate></item></items>
                <totalCount>2</totalCount></body></response>""".encode()
            return r

        with patch.dict(os.environ, {"SERVICE_KEY": "a%2Bb"}), \
                patch.object(api_client.requests, "get", side_effect=[page("A1"), page("A2")]) as get:
            self.assertEqual([r["hpid"] for r in api_client.fetch_national_beds()], ["A1", "A2"])
            self.assertEqual(get.call_args.kwargs["params"]["serviceKey"], "a+b")
        with patch.dict(os.environ, {"SERVICE_KEY": "a"}), \
                patch.object(api_client.requests, "get", side_effect=[page("A1"), page("A1")]):
            with self.assertRaisesRegex(RuntimeError, "nemc_repeated_page"):
                api_client.fetch_national_beds()


if __name__ == "__main__":
    unittest.main()

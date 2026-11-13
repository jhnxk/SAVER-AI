"""Serve one coherent hospital snapshot per request and refresh it off the request path."""
import json
import logging
import math
import os
from pathlib import Path
from threading import Lock, Thread
from time import perf_counter, sleep
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from api_client import fetch_national_beds
from update_realtime_resources_national import REALTIME_COLUMNS

ROOT = Path(__file__).resolve().parent
KST = ZoneInfo("Asia/Seoul")
OBSERVED_FIELDS = tuple(c for c in REALTIME_COLUMNS if c not in ("hpid", "data_fetched_at"))
logger = logging.getLogger(__name__)


class RefreshThrottled(Exception):
    """The previous refresh is still running or within its cooldown."""


def parse_observed(value):
    try:
        return datetime.strptime(str(value), "%Y%m%d%H%M%S").replace(tzinfo=KST)
    except ValueError:
        return None


def resource_value(field, value):
    if field.endswith("ayn"):
        return {"Y": "Y", "N": "N", "N1": "N"}.get(str(value).strip().upper(), "")
    try:
        number = float(value)
        return int(number) if math.isfinite(number) and number.is_integer() and number >= 0 else ""
    except (ValueError, TypeError):
        return ""


class LiveHospitalStore:
    def __init__(self, root=ROOT, fetch=fetch_national_beds, clock=None):
        self.root = root
        self.fetch = fetch
        self.clock = clock or (lambda: datetime.now(KST))
        self.interval = max(60, int(os.getenv("NEMC_REFRESH_SECONDS", "180")))
        self.max_age = timedelta(seconds=max(60, int(os.getenv("NEMC_MAX_AGE_SECONDS", "900"))))
        snapshot_dir = os.getenv("IEUM_SNAPSHOT_DIR")
        self.path = Path(snapshot_dir) / "nemc.json" if snapshot_dir else None
        self.lock = Lock()
        self.refresh_lock = Lock()
        self.baseline = None
        self.known_ids = set()
        self.minimum_coverage = 0
        self.observations = {}
        self.last_success = None
        self.last_attempt = None
        self.last_error = None
        self.refresh_ms = None
        self.last_attempt_monotonic = None
        self.thread = None

    def initialize(self):
        if self.baseline is not None:
            return
        base = pd.read_excel(
            self.root / "national_hospital_db_departments_hira_updated.xlsx",
            sheet_name="national_hospital_db", dtype={"encrypted_ykiho": str},
        ).fillna("")
        crosswalk = pd.read_csv(self.root / "data/nemc_hpid_crosswalk.csv", dtype=str)
        if not crosswalk["hpid"].str.fullmatch(r"[A-Za-z][0-9]+").all() or not crosswalk["hpid"].is_unique:
            raise ValueError("Invalid or duplicated HPID in the crosswalk")
        mapping = crosswalk.set_index("encrypted_ykiho", verify_integrity=True)["hpid"]
        if not base["encrypted_ykiho"].is_unique or not mapping.index.isin(base["encrypted_ykiho"]).all():
            raise ValueError("The crosswalk does not match the HIRA baseline")
        base["hpid"] = base["encrypted_ykiho"].map(mapping).fillna("")
        base[list(OBSERVED_FIELDS) + ["data_fetched_at", "api_matched_name"]] = ""
        self.known_ids = set(mapping)
        self.minimum_coverage = math.ceil(len(mapping) * 0.75)
        if self.path and self.path.exists():
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                seen = self.normalize(saved["rows"])
                if len(seen) >= self.minimum_coverage:
                    self.observations = seen
                    self.last_success = saved["fetched_at"]
            except (OSError, ValueError, KeyError, TypeError):
                logger.warning("Stored NEMC snapshot could not be loaded")
        self.baseline = base

    def normalize(self, rows):
        observations = {}
        for row in rows:
            hpid = str(row.get("hpid") or "").strip()
            observed = parse_observed(row.get("hvidate"))
            if hpid not in self.known_ids or observed is None:
                continue
            fields = {c: resource_value(c, row.get(c)) for c in OBSERVED_FIELDS if c != "hvidate"}
            if not any(value != "" for value in fields.values()):
                continue
            fields.update(hpid=hpid, hvidate=observed.strftime("%Y%m%d%H%M%S"))
            previous = observations.get(hpid)
            if previous and previous["hvidate"] == fields["hvidate"] and previous != fields:
                # Conflicting observations at the same source time cannot prove a resource is available.
                fields = {c: fields[c] if fields[c] == previous[c] else "" for c in fields}
            if not previous or fields["hvidate"] >= previous["hvidate"]:
                observations[hpid] = fields
        return observations

    def refresh(self, *, cooldown=0):
        if not self.refresh_lock.acquire(blocking=False):
            if cooldown:
                raise RefreshThrottled
            return False
        started = perf_counter()
        try:
            previous_attempt = self.last_attempt_monotonic
            if cooldown and previous_attempt is not None and started - previous_attempt < cooldown:
                raise RefreshThrottled
            self.last_attempt_monotonic = started
            now = self.clock()
            with self.lock:
                self.last_attempt = now.isoformat(timespec="seconds")
            incoming = self.normalize(self.fetch())
            if len(incoming) < self.minimum_coverage:
                raise RuntimeError(f"low_coverage_{len(incoming)}_of_{len(self.known_ids)}")
            fresh = sum(self.is_fresh(row, self.clock()) for row in incoming.values())
            if not fresh:
                raise RuntimeError("no_fresh_provider_observations")
            for hpid, row in incoming.items():
                previous = self.observations.get(hpid)
                if previous and previous["hvidate"] > row["hvidate"] and self.is_fresh(previous, self.clock()):
                    incoming[hpid] = previous
            fetched_at = self.clock().isoformat(timespec="seconds")
            if self.path:
                tmp = self.path.with_suffix(".tmp")
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    tmp.write_text(json.dumps({"fetched_at": fetched_at, "rows": list(incoming.values())}), encoding="utf-8")
                    tmp.replace(self.path)
                except OSError:
                    logger.warning("NEMC snapshot persistence failed")
            elapsed = round((perf_counter() - started) * 1000)
            with self.lock:
                self.observations = incoming
                self.last_success = fetched_at
                self.last_error = None
                self.refresh_ms = elapsed
            logger.info("NEMC refreshed mapped=%s fresh=%s elapsed_ms=%s", len(incoming), fresh, elapsed)
            return True
        except (RuntimeError, ValueError) as error:
            with self.lock:
                self.last_error = str(error)
                self.refresh_ms = round((perf_counter() - started) * 1000)
            logger.warning("NEMC refresh failed: %s", error)
            return False
        finally:
            self.refresh_lock.release()

    def is_fresh(self, row, now):
        observed = parse_observed(row["hvidate"])
        return observed is not None and -timedelta(minutes=2) <= now - observed <= self.max_age

    def start(self, *, automatic=True):
        self.initialize()
        if not automatic:
            return
        with self.lock:
            if self.thread is None:
                self.thread = Thread(target=self.run, name="nemc-refresh", daemon=True)
                self.thread.start()

    def run(self):
        while True:
            started = perf_counter()
            self.refresh()
            sleep(max(1, self.interval - (perf_counter() - started)))

    def snapshot(self):
        with self.lock:
            observations = self.observations
            meta = {
                "source": "nemc_background",
                "db_updated_at": self.last_success,
                "synced_at": self.last_success,
                "last_attempt_at": self.last_attempt,
                "last_error": self.last_error,
                "refresh_ms": self.refresh_ms,
            }
        now = self.clock()
        active = {key: row for key, row in observations.items() if self.is_fresh(row, now)}
        df = self.baseline.copy()
        df["realtime_match_status"] = df["hpid"].map(
            lambda key: "OK" if key in active else ("STALE" if key in observations else "UNVERIFIED")
        )
        if active:
            resources = pd.DataFrame.from_dict(active, orient="index")
            for field in OBSERVED_FIELDS:
                df[field] = df["hpid"].map(resources[field]).fillna("")
            df.loc[df["realtime_match_status"].eq("OK"), "data_fetched_at"] = meta["db_updated_at"]
        meta.update(
            status=("degraded" if meta["last_error"] else "fresh") if active else (
                "stale" if observations else ("unavailable" if meta["last_error"] else "warming")
            ),
            hospital_count=len(df),
            mapped_count=len(self.known_ids),
            observed_count=len(observations),
            fresh_count=len(active),
            provider_latest_at=max((parse_observed(row["hvidate"]).isoformat() for row in active.values()), default=None),
            freshness_limit_seconds=int(self.max_age.total_seconds()),
        )
        return df, meta

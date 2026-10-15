"""Download trials from ClinicalTrials.gov, parse them, split their criteria, store in Postgres.

    python -m trial_matcher.ingest [--max-trials N]

Safe to re-run: trials are upserted by nct_id and a trial's criteria rows are replaced.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from trial_matcher import config
from trial_matcher.ctgov_client import CTGovClient
from trial_matcher.db import PostgresStore, RecordError, connect
from trial_matcher.parse_trial import ParsedTrial, parse_trial
from trial_matcher.split_criteria import Criterion, split_criteria

log = logging.getLogger("trial_matcher.ingest")

MAX_SKIPPED_KEPT = 1000  # cap on skipped-record details stored in ingest_runs
MAX_PAGE_SIZE = 1000  # the API's limit


class Store(Protocol):
    def start_run(self, query: str) -> int: ...
    def update_run(self, run_id: int, **fields) -> None: ...
    def save_trial(self, trial: ParsedTrial, criteria: list[Criterion]) -> None: ...


@dataclass
class IngestStats:
    seen: int = 0
    stored: int = 0
    skipped: int = 0
    criteria: int = 0
    skipped_details: list[dict] = field(default_factory=list)
    split_notes: Counter = field(default_factory=Counter)

    def skip(self, nct_id: str | None, reason: str) -> None:
        self.skipped += 1
        if len(self.skipped_details) < MAX_SKIPPED_KEPT:
            self.skipped_details.append({"nct_id": nct_id, "reason": reason})
        log.warning("skipped %s: %s", nct_id or "<no id>", reason)


def _run_fields(stats: IngestStats, status: str = "running", error: str | None = None) -> dict:
    return {
        "trials_seen": stats.seen,
        "trials_stored": stats.stored,
        "trials_skipped": stats.skipped,
        "skipped": stats.skipped_details,
        "status": status,
        "error": error,
    }


def _ingest_study(study: dict, store: Store, stats: IngestStats) -> None:
    stats.seen += 1
    try:
        trial = parse_trial(study)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        nct = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
        stats.skip(nct, f"parse error: {exc}")
        return
    split = split_criteria(trial.eligibility_text)
    try:
        store.save_trial(trial, split.criteria)
    except RecordError as exc:
        stats.skip(trial.nct_id, f"database rejected record: {exc}")
        return
    stats.stored += 1
    stats.criteria += len(split.criteria)
    stats.split_notes.update(split.notes)


def run_ingest(client, store: Store, max_trials: int | None = None) -> IngestStats:
    stats = IngestStats()
    query = json.dumps({**client.query, "max_trials": max_trials}, sort_keys=True)
    run_id = store.start_run(query)
    total = client.count()
    if max_trials is not None:
        total = min(total, max_trials)
    log.info("run %s: fetching %d trials", run_id, total)
    try:
        for page_no, page in enumerate(client.iter_pages(max_trials), start=1):
            for study in page:
                _ingest_study(study, store, stats)
            store.update_run(run_id, **_run_fields(stats))
            log.info(
                "page %d: %d/%d seen, %d stored, %d skipped",
                page_no, stats.seen, total, stats.stored, stats.skipped,
            )
    except BaseException as exc:  # includes Ctrl-C: record the failure, then re-raise
        store.update_run(run_id, **_run_fields(stats, "failed", f"{type(exc).__name__}: {exc}"))
        log.error("run %s failed after %d trials: %s", run_id, stats.seen, exc)
        raise
    store.update_run(run_id, **_run_fields(stats, "done"))
    log.info(
        "run %s done: %d seen, %d stored (%d criteria), %d skipped",
        run_id, stats.seen, stats.stored, stats.criteria, stats.skipped,
    )
    if stats.split_notes:
        log.info("criteria splitter fallbacks: %s", dict(stats.split_notes.most_common()))
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--max-trials", type=int, default=None, help="stop after N trials (default: no cap)")
    parser.add_argument("--database-url", default=None, help="default: $DATABASE_URL")
    args = parser.parse_args(argv)
    if args.max_trials is not None and args.max_trials < 1:
        parser.error("--max-trials must be at least 1")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per request is noise
    page_size = MAX_PAGE_SIZE if args.max_trials is None else min(MAX_PAGE_SIZE, args.max_trials)
    client = CTGovClient(page_size=page_size)
    with connect(args.database_url or config.database_url()) as conn:
        stats = run_ingest(client, PostgresStore(conn), args.max_trials)
    return 0 if stats.seen > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

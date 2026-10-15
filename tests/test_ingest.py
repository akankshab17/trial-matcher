"""Ingest logic against a fake API client and a fake store (no network, no database)."""

import copy
import json
import logging
from pathlib import Path

import pytest

from trial_matcher import ingest
from trial_matcher.db import RecordError
from trial_matcher.ingest import run_ingest

FIXTURES = Path(__file__).parent / "fixtures"
GOOD = json.loads((FIXTURES / "studies_parse_cases.json").read_text())["studies"]  # 5 real trials


def study(nct, title="T", eligibility=None):
    s = {"protocolSection": {"identificationModule": {"nctId": nct, "briefTitle": title}}}
    if eligibility is not None:
        s["protocolSection"]["eligibilityModule"] = {"eligibilityCriteria": eligibility}
    return s


class FakeClient:
    def __init__(self, pages, total=None, fail_after_pages=None):
        self.pages = pages
        self.query = {"query.cond": "cancer"}
        self.total = total if total is not None else sum(len(p) for p in pages)
        self.fail_after_pages = fail_after_pages
        self.max_trials_seen = "unset"

    def count(self):
        return self.total

    def iter_pages(self, max_trials=None):
        self.max_trials_seen = max_trials
        left = max_trials
        for i, page in enumerate(self.pages):
            if self.fail_after_pages is not None and i >= self.fail_after_pages:
                raise RuntimeError("API went away")
            page = page if left is None else page[:left]
            if left is not None:
                left -= len(page)
            if page:
                yield copy.deepcopy(page)
            if left == 0:
                return


class FakeStore:
    def __init__(self, reject=()):
        self.reject = set(reject)
        self.saved = {}  # nct_id -> (trial, criteria)
        self.runs = {}
        self.updates = []

    def start_run(self, query):
        self.runs[1] = {"query": query}
        return 1

    def update_run(self, run_id, **fields):
        self.runs[run_id].update(fields)
        self.updates.append(fields)

    def save_trial(self, trial, criteria):
        if trial.nct_id in self.reject:
            raise RecordError("DataError: nope")
        self.saved[trial.nct_id] = (trial, criteria)


def test_stores_every_trial_and_its_criteria():
    store = FakeStore()
    stats = run_ingest(FakeClient([GOOD[:3], GOOD[3:]]), store)
    assert (stats.seen, stats.stored, stats.skipped) == (5, 5, 0)
    assert set(store.saved) == {s["protocolSection"]["identificationModule"]["nctId"] for s in GOOD}
    trial, criteria = store.saved["NCT06647862"]
    assert trial.min_age_years == 18.0
    assert len(criteria) > 5 and criteria[0].kind == "inclusion"
    assert stats.criteria == sum(len(c) for _, c in store.saved.values())


def test_bad_records_are_skipped_and_logged_not_fatal(caplog):
    pages = [[study("NCT1", eligibility="Inclusion Criteria:\n* a"), {"protocolSection": {}}, study("NCT3")]]
    pages[0].append(study("NCT4", title="x"))
    store = FakeStore(reject={"NCT4"})
    with caplog.at_level(logging.WARNING):
        stats = run_ingest(FakeClient(pages), store)
    assert (stats.seen, stats.stored, stats.skipped) == (4, 2, 2)
    assert set(store.saved) == {"NCT1", "NCT3"}
    reasons = {d["nct_id"]: d["reason"] for d in stats.skipped_details}
    assert "parse error" in reasons[None]
    assert "database rejected" in reasons["NCT4"]
    assert "NCT4" in caplog.text


def test_trial_with_no_eligibility_text_is_stored_with_zero_criteria():
    store = FakeStore()
    stats = run_ingest(FakeClient([[study("NCT1")]]), store)
    assert stats.stored == 1 and store.saved["NCT1"][1] == []


def test_run_is_logged_start_progress_and_done():
    store = FakeStore()
    run_ingest(FakeClient([GOOD[:2], GOOD[2:]]), store, max_trials=None)
    assert json.loads(store.runs[1]["query"])["query.cond"] == "cancer"
    assert [u["trials_seen"] for u in store.updates] == [2, 5, 5]  # one per page, then final
    assert store.runs[1]["status"] == "done"
    assert store.runs[1]["trials_stored"] == 5 and store.runs[1]["trials_skipped"] == 0
    assert store.runs[1]["error"] is None


def test_skipped_details_are_saved_on_the_run():
    store = FakeStore()
    run_ingest(FakeClient([[{"protocolSection": {}}]]), store)
    assert store.runs[1]["trials_skipped"] == 1
    assert store.runs[1]["skipped"][0]["reason"].startswith("parse error")


def test_progress_is_logged_every_page(caplog):
    with caplog.at_level(logging.INFO, logger="trial_matcher.ingest"):
        run_ingest(FakeClient([GOOD[:2], GOOD[2:4], GOOD[4:]]), FakeStore())
    page_lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("page ")]
    assert page_lines == [
        "page 1: 2/5 seen, 2 stored, 0 skipped",
        "page 2: 4/5 seen, 4 stored, 0 skipped",
        "page 3: 5/5 seen, 5 stored, 0 skipped",
    ]


def test_max_trials_is_passed_to_the_client_and_caps_the_total():
    client = FakeClient([GOOD[:3], GOOD[3:]])
    store = FakeStore()
    stats = run_ingest(client, store, max_trials=4)
    assert client.max_trials_seen == 4 and stats.seen == 4
    assert json.loads(store.runs[1]["query"])["max_trials"] == 4


def test_no_cap_by_default():
    client = FakeClient([GOOD])
    run_ingest(client, FakeStore())
    assert client.max_trials_seen is None


def test_crash_marks_run_failed_and_reraises():
    store = FakeStore()
    with pytest.raises(RuntimeError, match="API went away"):
        run_ingest(FakeClient([GOOD[:2], GOOD[2:]], fail_after_pages=1), store)
    assert store.runs[1]["status"] == "failed"
    assert "API went away" in store.runs[1]["error"]
    assert store.runs[1]["trials_stored"] == 2  # what was saved before the crash is kept


def test_ctrl_c_marks_run_failed():
    class Interrupted(FakeClient):
        def iter_pages(self, max_trials=None):
            raise KeyboardInterrupt
            yield

    store = FakeStore()
    with pytest.raises(KeyboardInterrupt):
        run_ingest(Interrupted([]), store)
    assert store.runs[1]["status"] == "failed"


def test_splitter_fallbacks_are_counted():
    stats = run_ingest(FakeClient([[study("NCT1", eligibility="* a\n* b")]]), FakeStore())
    assert stats.split_notes["no_headers"] == 1


def test_main_wires_max_trials_into_client_page_size(monkeypatch):
    made = {}

    class StubClient(FakeClient):
        def __init__(self, page_size):
            super().__init__([GOOD])
            made["page_size"] = page_size

    class StubConn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(ingest, "CTGovClient", StubClient)
    monkeypatch.setattr(ingest, "connect", lambda url: StubConn())
    monkeypatch.setattr(ingest, "PostgresStore", lambda conn: FakeStore())
    assert ingest.main(["--max-trials", "100", "--database-url", "postgresql://x"]) == 0
    assert made["page_size"] == 100
    ingest.main(["--database-url", "postgresql://x"])
    assert made["page_size"] == 1000


def test_main_rejects_nonpositive_max_trials():
    with pytest.raises(SystemExit):
        ingest.main(["--max-trials", "0", "--database-url", "postgresql://x"])

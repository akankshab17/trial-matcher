"""PostgresStore and the full ingest against a real (throwaway) Postgres. See conftest.py."""

import json
from pathlib import Path

import pytest

from tests.test_ingest import FakeClient, study
from trial_matcher.db import PostgresStore, RecordError
from trial_matcher.ingest import run_ingest
from trial_matcher.parse_trial import parse_trial
from trial_matcher.split_criteria import split_criteria

GOOD = json.loads((Path(__file__).parent / "fixtures" / "studies_parse_cases.json").read_text())["studies"]
ELIG = "Inclusion Criteria:\n* Age 18+\n* ECOG 0-1\nExclusion Criteria:\n* Pregnant"


def save(store, s):
    trial = parse_trial(s)
    store.save_trial(trial, split_criteria(trial.eligibility_text).criteria)
    return trial


def count(conn, table, where="", args=()):
    return conn.execute(f"SELECT count(*) FROM {table} {where}", args).fetchone()[0]


def test_migrations_create_the_tables(conn):
    names = {r[0] for r in conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()")}
    assert {"trials", "criteria", "ingest_runs"} <= names


def test_save_trial_writes_trial_and_criteria(conn):
    store = PostgresStore(conn)
    save(store, GOOD[1])  # NCT07118566: 18-75 years
    row = conn.execute(
        "SELECT title, status, phase, min_age_years, max_age_years, healthy_volunteers, raw_json, last_update_posted "
        "FROM trials WHERE nct_id = 'NCT07118566'"
    ).fetchone()
    assert row[2] == ["PHASE2"] and (row[3], row[4]) == (18.0, 75.0) and row[5] is False
    assert row[6]["protocolSection"]["identificationModule"]["nctId"] == "NCT07118566"  # jsonb round trip
    assert row[7] is not None
    assert count(conn, "criteria", "WHERE nct_id = 'NCT07118566'") > 3


def test_null_ages_are_stored_as_null(conn):
    save(PostgresStore(conn), GOOD[4])  # NCT07850154: no age limits
    assert conn.execute("SELECT min_age_years, max_age_years FROM trials").fetchone() == (None, None)


def test_rerun_does_not_duplicate_trials_or_criteria(conn):
    store = PostgresStore(conn)
    for _ in range(3):
        for s in GOOD:
            save(store, s)
    assert count(conn, "trials") == 5
    per_trial = conn.execute("SELECT nct_id, count(*), count(DISTINCT position) FROM criteria GROUP BY nct_id").fetchall()
    assert all(n == distinct for _, n, distinct in per_trial)
    first = count(conn, "criteria")
    save(store, GOOD[0])
    assert count(conn, "criteria") == first


def test_reingest_replaces_changed_trial_and_its_criteria(conn):
    store = PostgresStore(conn)
    save(store, study("NCT1", title="Old title", eligibility=ELIG))
    assert count(conn, "criteria") == 3
    save(store, study("NCT1", title="New title", eligibility="Inclusion Criteria:\n* Only one now"))
    assert conn.execute("SELECT title FROM trials").fetchone()[0] == "New title"
    assert [r[0] for r in conn.execute("SELECT text FROM criteria")] == ["Only one now"]


def test_reingest_with_no_eligibility_text_clears_old_criteria(conn):
    store = PostgresStore(conn)
    save(store, study("NCT1", eligibility=ELIG))
    save(store, study("NCT1"))
    assert count(conn, "trials") == 1 and count(conn, "criteria") == 0


def test_other_trials_criteria_are_untouched_by_a_reingest(conn):
    store = PostgresStore(conn)
    save(store, study("NCT1", eligibility=ELIG))
    save(store, study("NCT2", eligibility=ELIG))
    save(store, study("NCT1", eligibility="Inclusion Criteria:\n* x"))
    assert count(conn, "criteria", "WHERE nct_id = 'NCT2'") == 3


def test_bad_record_raises_record_error_and_leaves_no_partial_data(conn):
    store = PostgresStore(conn)
    with pytest.raises(RecordError):
        save(store, study("NCT1", title="bad \x00 title", eligibility=ELIG))  # NUL is illegal in Postgres text
    assert count(conn, "trials") == 0 and count(conn, "criteria") == 0
    save(store, study("NCT2", eligibility=ELIG))  # connection still usable
    assert count(conn, "trials") == 1


def test_failed_reingest_keeps_the_old_version(conn):
    store = PostgresStore(conn)
    save(store, study("NCT1", title="Good", eligibility=ELIG))
    with pytest.raises(RecordError):
        save(store, study("NCT1", title="Bad \x00", eligibility="Inclusion Criteria:\n* x"))
    assert conn.execute("SELECT title FROM trials").fetchone()[0] == "Good"
    assert count(conn, "criteria") == 3


def test_run_lifecycle(conn):
    store = PostgresStore(conn)
    run_id = store.start_run('{"q": 1}')
    status, finished = conn.execute("SELECT status, finished_at FROM ingest_runs WHERE id = %s", (run_id,)).fetchone()
    assert status == "running" and finished is None
    store.update_run(run_id, trials_seen=5, trials_stored=4, trials_skipped=1,
                     skipped=[{"nct_id": "NCT9", "reason": "x"}], status="done")
    row = conn.execute("SELECT status, finished_at, trials_seen, trials_stored, trials_skipped, skipped, error "
                       "FROM ingest_runs WHERE id = %s", (run_id,)).fetchone()
    assert row[0] == "done" and row[1] is not None and row[2:5] == (5, 4, 1)
    assert row[5] == [{"nct_id": "NCT9", "reason": "x"}] and row[6] is None


def test_failed_run_keeps_its_error(conn):
    store = PostgresStore(conn)
    run_id = store.start_run("{}")
    store.update_run(run_id, trials_seen=0, trials_stored=0, trials_skipped=0, skipped=[],
                     status="failed", error="RuntimeError: boom")
    assert conn.execute("SELECT status, error FROM ingest_runs").fetchone() == ("failed", "RuntimeError: boom")


def test_full_ingest_end_to_end_is_idempotent(conn):
    pages = [GOOD[:2], GOOD[2:], [{"protocolSection": {}}]]  # includes one bad record
    for _ in range(2):
        stats = run_ingest(FakeClient(pages), PostgresStore(conn))
        assert (stats.seen, stats.stored, stats.skipped) == (6, 5, 1)
    assert count(conn, "trials") == 5
    assert count(conn, "ingest_runs") == 2
    assert conn.execute("SELECT status, trials_stored, trials_skipped FROM ingest_runs ORDER BY id DESC").fetchone() == ("done", 5, 1)
    assert count(conn, "criteria") == stats.criteria


# --- full-text column added in migration 003 --------------------------------


def fts_ids(conn, query):
    return [r[0] for r in conn.execute(
        "SELECT nct_id FROM trials WHERE search_tsv @@ to_tsquery('english', %s) ORDER BY nct_id", (query,))]


def test_search_tsv_covers_title_conditions_and_interventions_with_stemming(conn):
    store = PostgresStore(conn)
    s = study("NCT1", title="A phase 2 study")
    s["protocolSection"]["conditionsModule"] = {"conditions": ["Breast Neoplasms"]}
    s["protocolSection"]["armsInterventionsModule"] = {"interventions": [{"name": "Trastuzumab"}]}
    save(store, s)
    save(store, study("NCT2", title="Melanoma immunotherapy"))
    assert fts_ids(conn, "neoplasm") == ["NCT1"]  # condition, stemmed
    assert fts_ids(conn, "trastuzumab") == ["NCT1"]  # intervention
    assert fts_ids(conn, "melanoma | trastuzumab") == ["NCT1", "NCT2"]  # OR query
    assert fts_ids(conn, "pancreatic") == []


def test_search_tsv_updates_when_a_trial_is_reingested(conn):
    store = PostgresStore(conn)
    save(store, study("NCT1", title="Old topic"))
    save(store, study("NCT1", title="Glioblastoma"))
    assert fts_ids(conn, "glioblastoma") == ["NCT1"] and fts_ids(conn, "topic") == []

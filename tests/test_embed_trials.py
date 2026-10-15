"""Embedding run against a real throwaway Postgres (see conftest.py) with a fake encoder."""

import hashlib
import logging
import math

import pytest

from tests.test_ingest import study
from trial_matcher import embed_trials
from trial_matcher.db import PostgresStore
from trial_matcher.embed import BASIC, WITH_INCLUSION
from trial_matcher.embed_trials import run_embedding, summarize
from trial_matcher.parse_trial import parse_trial
from trial_matcher.split_criteria import split_criteria

MODEL = "fake/model-a"


class FakeEncoder:
    """Deterministic 8-d unit vectors derived from the text; records what it was asked to embed."""

    def __init__(self, dim=8):
        self.dim = dim
        self.seen = []

    def encode(self, texts):
        self.seen.extend(texts)
        out = []
        for t in texts:
            raw = [b + 1 for b in hashlib.sha256(t.encode()).digest()[: self.dim]]
            norm = math.sqrt(sum(x * x for x in raw))
            out.append([x / norm for x in raw])
        return out


def add_trial(conn, nct, title="A trial", conditions=("Breast Cancer",), interventions=("Drug X",), eligibility=None):
    s = study(nct, title=title, eligibility=eligibility)
    s["protocolSection"]["conditionsModule"] = {"conditions": list(conditions)}
    s["protocolSection"]["armsInterventionsModule"] = {"interventions": [{"name": i} for i in interventions]}
    trial = parse_trial(s)
    PostgresStore(conn).save_trial(trial, split_criteria(trial.eligibility_text).criteria)


def rows(conn, model=MODEL, variant=BASIC):
    return conn.execute(
        "SELECT nct_id, dim, text_hash FROM trial_embeddings WHERE model = %s AND variant = %s ORDER BY nct_id",
        (model, variant),
    ).fetchall()


@pytest.fixture
def three_trials(conn):
    for n in ("NCT1", "NCT2", "NCT3"):
        add_trial(conn, n, title=f"Trial {n}")
    return conn


def test_embeds_every_trial(three_trials):
    stats = run_embedding(three_trials, FakeEncoder(), MODEL)
    assert (stats.trials, stats.embedded, stats.unchanged) == (3, 3, 0)
    assert [(r[0], r[1]) for r in rows(three_trials)] == [("NCT1", 8), ("NCT2", 8), ("NCT3", 8)]


def test_stored_vector_round_trips_and_is_searchable(three_trials):
    enc = FakeEncoder()
    run_embedding(three_trials, enc, MODEL)
    [target] = enc.encode(["Trial NCT2. Conditions: Breast Cancer. Interventions: Drug X"])
    nearest = three_trials.execute(
        "SELECT nct_id FROM trial_embeddings WHERE model = %s ORDER BY embedding <=> %s::vector LIMIT 1",
        (MODEL, str(target)),
    ).fetchone()
    assert nearest[0] == "NCT2"


def test_rerun_skips_unchanged_trials(three_trials):
    run_embedding(three_trials, FakeEncoder(), MODEL)
    enc = FakeEncoder()
    stats = run_embedding(three_trials, enc, MODEL)
    assert (stats.embedded, stats.unchanged) == (0, 3)
    assert enc.seen == []  # the model was not even asked
    assert len(rows(three_trials)) == 3


def test_only_the_changed_trial_is_reembedded(three_trials):
    run_embedding(three_trials, FakeEncoder(), MODEL)
    before = dict((r[0], r[2]) for r in rows(three_trials))
    add_trial(three_trials, "NCT2", title="Trial NCT2 renamed")  # re-ingest with a new title
    enc = FakeEncoder()
    stats = run_embedding(three_trials, enc, MODEL)
    assert (stats.embedded, stats.unchanged) == (1, 2)
    assert len(enc.seen) == 1 and "renamed" in enc.seen[0]
    after = dict((r[0], r[2]) for r in rows(three_trials))
    assert after["NCT2"] != before["NCT2"] and after["NCT1"] == before["NCT1"]
    assert len(rows(three_trials)) == 3  # replaced, not duplicated


def test_models_and_variants_are_stored_side_by_side(three_trials):
    run_embedding(three_trials, FakeEncoder(8), "fake/model-a")
    run_embedding(three_trials, FakeEncoder(16), "fake/model-b")  # different dimension
    run_embedding(three_trials, FakeEncoder(8), "fake/model-a", WITH_INCLUSION)
    assert len(rows(three_trials, "fake/model-a", BASIC)) == 3
    assert {r[1] for r in rows(three_trials, "fake/model-b")} == {16}
    assert len(rows(three_trials, "fake/model-a", WITH_INCLUSION)) == 3
    assert three_trials.execute("SELECT count(*) FROM trial_embeddings").fetchone()[0] == 9


def test_with_inclusion_variant_embeds_the_first_inclusion_criteria(conn):
    add_trial(conn, "NCT1", eligibility="Inclusion Criteria:\n* HER2-negative\n* Stage IV\nExclusion Criteria:\n* Pregnant")
    enc = FakeEncoder()
    run_embedding(conn, enc, MODEL, WITH_INCLUSION)
    assert enc.seen[0].endswith("Inclusion: HER2-negative Stage IV")
    assert "Pregnant" not in enc.seen[0]


def test_limit_only_embeds_the_first_n_by_nct_id(three_trials):
    stats = run_embedding(three_trials, FakeEncoder(), MODEL, limit=2)
    assert stats.trials == 2
    assert [r[0] for r in rows(three_trials)] == ["NCT1", "NCT2"]


def test_chunks_commit_separately_and_log_progress(three_trials, caplog):
    with caplog.at_level(logging.INFO, logger="trial_matcher.embed_trials"):
        run_embedding(three_trials, FakeEncoder(), MODEL, chunk_size=2)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("embedded")]
    assert lines == ["embedded 2/3", "embedded 3/3"]


def test_interrupted_run_keeps_finished_chunks_and_resumes(three_trials):
    class Dies(FakeEncoder):
        def encode(self, texts):
            if self.seen:
                raise KeyboardInterrupt
            return super().encode(texts)

    with pytest.raises(KeyboardInterrupt):
        run_embedding(three_trials, Dies(), MODEL, chunk_size=2)
    assert len(rows(three_trials)) == 2  # first chunk survived
    stats = run_embedding(three_trials, FakeEncoder(), MODEL, chunk_size=2)
    assert (stats.embedded, stats.unchanged) == (1, 2)


def test_encoder_returning_wrong_count_is_an_error(three_trials):
    class Broken(FakeEncoder):
        def encode(self, texts):
            return super().encode(texts)[:-1]

    with pytest.raises(RuntimeError, match="vectors for"):
        run_embedding(three_trials, Broken(), MODEL)
    assert rows(three_trials) == []


def test_deleting_a_trial_deletes_its_embeddings(three_trials):
    run_embedding(three_trials, FakeEncoder(), MODEL)
    three_trials.execute("DELETE FROM trials WHERE nct_id = 'NCT1'")
    assert [r[0] for r in rows(three_trials)] == ["NCT2", "NCT3"]


def test_empty_database_embeds_nothing(conn):
    stats = run_embedding(conn, FakeEncoder(), MODEL)
    assert (stats.trials, stats.embedded) == (0, 0)


def test_summarize_reports_counts_and_missing(three_trials):
    run_embedding(three_trials, FakeEncoder(), MODEL, limit=2)
    assert summarize(three_trials, MODEL, BASIC) == {
        "trials": 3, "embedded": 2, "missing": 1, "distinct_dims": 1, "dim": 8,
    }


def test_main_uses_model_variant_and_limit_options(conn, pg_url, monkeypatch):
    # main() opens its own connection, so point it at the schema this test's `conn` is using.
    schema = conn.execute("SELECT current_schema()").fetchone()[0]
    add_trial(conn, "NCT1")
    add_trial(conn, "NCT2")
    loaded = []
    monkeypatch.setattr(embed_trials, "SentenceTransformerEncoder", lambda name: loaded.append(name) or FakeEncoder())
    monkeypatch.setattr(embed_trials, "connect", lambda url: __import__("psycopg").connect(
        url, autocommit=True, options=f"-c search_path={schema},public"))
    code = embed_trials.main(["--model", "my/model", "--variant", "with_inclusion", "--limit", "1",
                              "--database-url", pg_url])
    assert code == 0 and loaded == ["my/model"]
    assert [r[0] for r in rows(conn, "my/model", WITH_INCLUSION)] == ["NCT1"]


def test_main_refuses_an_empty_database_without_loading_the_model(conn, pg_url, monkeypatch):
    schema = conn.execute("SELECT current_schema()").fetchone()[0]
    monkeypatch.setattr(embed_trials, "SentenceTransformerEncoder", lambda name: pytest.fail("model loaded"))
    monkeypatch.setattr(embed_trials, "connect", lambda url: __import__("psycopg").connect(
        url, autocommit=True, options=f"-c search_path={schema},public"))
    assert embed_trials.main(["--database-url", pg_url]) == 1


@pytest.mark.parametrize("args", [["--limit", "0"], ["--chunk-size", "0"], ["--variant", "nope"]])
def test_main_rejects_bad_options(args):
    with pytest.raises(SystemExit):
        embed_trials.main([*args, "--database-url", "postgresql://x"])

"""Search against a real throwaway Postgres (see conftest.py) with hand-placed 3-d vectors."""

import logging

import pytest

from trial_matcher import search as search_module
from trial_matcher.db import PostgresStore
from trial_matcher.embed import BASIC, WITH_INCLUSION, vector_literal
from trial_matcher.parse_trial import parse_trial
from trial_matcher.search import (
    DISCLAIMER,
    SearchError,
    fts_query_text,
    query_prefix,
    reciprocal_rank_fusion,
    search,
)

MODEL = "fake/model"
X, Y, Z = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]


class QueryEncoder:
    """Always returns the same query vector; remembers the text it was given."""

    def __init__(self, vector=X):
        self.vector = vector
        self.texts = []

    def encode(self, texts):
        self.texts.extend(texts)
        return [self.vector for _ in texts]


def add(conn, nct, title, vec, *, conditions=(), interventions=(), status="RECRUITING",
        sex="ALL", min_age="18 Years", max_age=None, model=MODEL, variant=BASIC):
    """Store a trial (through the real parser/saver) and its embedding."""
    elig = {"sex": sex}
    if min_age:
        elig["minimumAge"] = min_age
    if max_age:
        elig["maximumAge"] = max_age
    if sex is None:
        del elig["sex"]
    study = {"protocolSection": {
        "identificationModule": {"nctId": nct, "briefTitle": title},
        "statusModule": {"overallStatus": status},
        "conditionsModule": {"conditions": list(conditions)},
        "armsInterventionsModule": {"interventions": [{"name": i} for i in interventions]},
        "eligibilityModule": elig,
    }}
    PostgresStore(conn).save_trial(parse_trial(study), [])
    if vec is not None:
        conn.execute(
            "INSERT INTO trial_embeddings (nct_id, model, variant, text_hash, embedding) VALUES (%s, %s, %s, 'h', %s::vector)",
            (nct, model, variant, vector_literal(vec)),
        )


def ids(results):
    return [r.nct_id for r in results]


def run(conn, query="tumor", age=50, sex="FEMALE", **kw):
    kw.setdefault("encoder", QueryEncoder())
    kw.setdefault("model", MODEL)
    return search(conn, query, age=age, sex=sex, **kw)


# --- pure helpers ----------------------------------------------------------------


def test_rrf_scores_and_order():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60)
    scores = dict(fused)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert [i for i, _ in fused] == ["b", "a", "d", "c"]  # b is in both lists; a=1/61 > d=1/62 > c=1/63


def test_rrf_ties_break_on_id_and_empty_input():
    assert [i for i, _ in reciprocal_rank_fusion([["z"], ["a"]])] == ["a", "z"]
    assert reciprocal_rank_fusion([[], []]) == []


def test_fts_query_text_is_an_or_of_clean_words():
    assert fts_query_text("58-year-old woman with HER2+ breast cancer") == (
        "58 or year or old or woman or with or her2 or breast or cancer")
    assert fts_query_text("cancer cancer Cancer") == "cancer"


def test_fts_query_text_neutralises_operators_and_the_word_or():
    assert fts_query_text('lung "cancer" -smoker | (a) & b; DROP TABLE trials; -- or') == (
        "lung or cancer or smoker or drop or table or trials")
    assert fts_query_text("!!! ?? a") == ""


def test_bge_models_get_the_query_prefix_others_do_not():
    assert query_prefix("BAAI/bge-small-en-v1.5").startswith("Represent this sentence")
    assert query_prefix("sentence-transformers/all-MiniLM-L6-v2") == ""


# --- hard filters -----------------------------------------------------------------


@pytest.fixture
def filter_world(conn):
    """Each trial is a perfect match for the query in both rankings; only the filters differ."""
    add(conn, "OK", "tumor trial", X)
    add(conn, "NYR", "tumor not yet recruiting", X, status="NOT_YET_RECRUITING")
    add(conn, "DONE", "tumor completed", X, status="COMPLETED")
    add(conn, "CLOSED", "tumor not recruiting", X, status="ACTIVE_NOT_RECRUITING")
    add(conn, "FEM", "tumor women only", X, sex="FEMALE")
    add(conn, "MALE", "tumor men only", X, sex="MALE")
    add(conn, "NOSEX", "tumor sex unknown", X, sex=None)
    add(conn, "KIDS", "tumor children", X, min_age="1 Year", max_age="17 Years")
    add(conn, "OLD", "tumor over sixty five", X, min_age="65 Years")
    add(conn, "NOLIMITS", "tumor no age limits", X, min_age=None)
    add(conn, "MONTHS", "tumor infants", X, min_age="6 Months", max_age="2 Years")
    return conn


@pytest.mark.parametrize("mode", ["hybrid", "vector", "fts"])
def test_status_filter_keeps_only_recruiting_and_not_yet_recruiting(filter_world, mode):
    got = set(ids(run(filter_world, mode=mode)))
    assert {"OK", "NYR"} <= got
    assert not got & {"DONE", "CLOSED"}


@pytest.mark.parametrize("mode", ["hybrid", "vector", "fts"])
def test_sex_filter(filter_world, mode):
    woman = set(ids(run(filter_world, sex="FEMALE", mode=mode)))
    man = set(ids(run(filter_world, sex="MALE", mode=mode)))
    assert {"FEM", "NOSEX", "OK"} <= woman and "MALE" not in woman
    assert {"MALE", "NOSEX", "OK"} <= man and "FEM" not in man


def test_sex_input_is_case_insensitive(filter_world):
    assert ids(run(filter_world, sex="female")) == ids(run(filter_world, sex="FEMALE"))


@pytest.mark.parametrize("mode", ["hybrid", "vector", "fts"])
def test_age_filter(filter_world, mode):
    adult = set(ids(run(filter_world, age=50, mode=mode)))
    assert {"OK", "NOLIMITS"} <= adult
    assert not adult & {"KIDS", "OLD", "MONTHS"}
    senior = set(ids(run(filter_world, age=70, mode=mode)))
    assert {"OK", "OLD", "NOLIMITS"} <= senior and "KIDS" not in senior
    child = set(ids(run(filter_world, age=10, mode=mode)))
    assert child == {"KIDS", "NOLIMITS"}
    baby = set(ids(run(filter_world, age=1.0, mode=mode)))  # 6 months..2 years, converted to years
    assert baby == {"KIDS", "MONTHS", "NOLIMITS"}  # KIDS starts at exactly 1 year; MONTHS is 0.5-2 years


def test_age_limits_are_inclusive(conn):
    add(conn, "A", "tumor", X, min_age="18 Years", max_age="65 Years")
    assert ids(run(conn, age=18)) == ["A"]
    assert ids(run(conn, age=65)) == ["A"]
    assert ids(run(conn, age=17.9)) == []
    assert ids(run(conn, age=65.1)) == []


def test_filters_apply_before_ranking_so_ineligible_perfect_matches_do_not_crowd_the_top(conn):
    for i in range(5):
        add(conn, f"KID{i}", "tumor tumor tumor", X, min_age=None, max_age="10 Years")  # perfect, but a child
    add(conn, "ADULT", "something else", [0.5, 0.5, 0.0])
    assert ids(run(conn, age=40, k=1, mode="vector")) == ["ADULT"]


# --- ranking and merging -------------------------------------------------------------


def test_vector_mode_orders_by_cosine_distance_and_reports_it(conn):
    add(conn, "NEAR", "alpha", [1.0, 0.1, 0.0])
    add(conn, "MID", "beta", [1.0, 1.0, 0.0])
    add(conn, "FAR", "gamma", Z)
    got = run(conn, mode="vector")
    assert ids(got) == ["NEAR", "MID", "FAR"]
    assert [r.vector_rank for r in got] == [1, 2, 3]
    assert got[0].vector_distance < got[1].vector_distance < got[2].vector_distance
    assert all(r.fts_rank is None for r in got)
    assert got[0].score == pytest.approx(1 / 61)


def test_fts_mode_matches_any_word_with_stemming_and_prefers_more_matches(conn):
    add(conn, "BOTH", "melanoma and glioma tumors", Z)
    add(conn, "ONE", "glioma", Z)
    add(conn, "NONE", "knee surgery", Z)
    got = run(conn, query="a patient with melanoma or glioma tumor", mode="fts")
    assert ids(got) == ["BOTH", "ONE"]  # 'tumor' matches 'tumors' (stemming); NONE is absent
    assert all(r.vector_rank is None for r in got)


def test_fts_searches_conditions_and_interventions_too(conn):
    add(conn, "C", "plain title", Z, conditions=["Pancreatic Neoplasms"])
    add(conn, "I", "plain title two", Z, interventions=["Gemcitabine"])
    assert set(ids(run(conn, query="gemcitabine neoplasm", mode="fts"))) == {"C", "I"}


def test_hybrid_merges_both_lists(conn):
    # vector order: V1 > V2 > B > T1; text order: B (2 words) > T1 > T2
    add(conn, "V1", "irrelevant a", X)
    add(conn, "V2", "irrelevant b", [1.0, 0.2, 0.0])
    add(conn, "B", "lung tumor", [1.0, 0.5, 0.0])
    add(conn, "T1", "lung", [0.0, 1.0, 0.0])
    add(conn, "T2", "tumor", None)  # no vector at all: found only by text
    got = run(conn, query="lung tumor")
    by = {r.nct_id: r for r in got}
    assert got[0].nct_id == "B"  # present in both lists beats the vector-only leaders
    assert by["B"].vector_rank == 3 and by["B"].fts_rank == 1
    assert by["B"].score == pytest.approx(1 / 63 + 1 / 61)
    assert by["V1"].fts_rank is None and by["T2"].vector_rank is None
    assert set(by) == {"V1", "V2", "B", "T1", "T2"}


def test_query_with_no_usable_words_still_works_on_vectors(conn):
    add(conn, "A", "tumor", X)
    got = run(conn, query="?! a")
    assert ids(got) == ["A"] and got[0].fts_rank is None


def test_hostile_query_text_is_harmless(conn):
    add(conn, "A", "tumor", X)
    run(conn, query="'; DROP TABLE trials; -- \" ( ) | & ! :* <->")
    assert conn.execute("SELECT count(*) FROM trials").fetchone()[0] == 1


def test_k_limits_and_pool_limits_results(conn):
    for i in range(6):
        add(conn, f"T{i}", "tumor", [1.0, i * 0.1, 0.0])
    assert len(run(conn, k=3)) == 3
    assert len(run(conn, k=100)) == 6
    assert len(run(conn, k=100, pool=2, mode="vector")) == 2


def test_default_k_is_50(conn):
    for i in range(55):
        add(conn, f"T{i:02d}", "tumor", [1.0, i * 0.01, 0.0])
    assert len(run(conn)) == 50


def test_results_are_deterministic_and_ties_break_on_nct_id(conn):
    for n in ("N3", "N1", "N2"):
        add(conn, n, "same", X)
    assert ids(run(conn, mode="vector")) == ["N1", "N2", "N3"]


def test_result_carries_trial_details(conn):
    add(conn, "A", "Drug X for lung cancer", X, conditions=["Lung Cancer"], status="NOT_YET_RECRUITING")
    [r] = run(conn)
    assert (r.title, r.status, r.conditions) == ("Drug X for lung cancer", "NOT_YET_RECRUITING", ["Lung Cancer"])


def test_no_candidates_gives_empty_list(conn):
    add(conn, "A", "tumor", X, status="COMPLETED")  # embedded, but filtered out
    assert run(conn) == []
    assert run(conn, query="zebra", mode="fts") == []


def test_empty_database_says_to_embed_first(conn):
    with pytest.raises(SearchError, match="embed_trials"):
        run(conn)


# --- model / variant handling --------------------------------------------------------


def test_only_the_requested_model_and_variant_are_searched(conn):
    add(conn, "A", "alpha", X)
    conn.execute("INSERT INTO trial_embeddings (nct_id, model, variant, text_hash, embedding) VALUES ('A', 'other', 'basic', 'h', '[0,0,1,0]')")
    conn.execute("INSERT INTO trial_embeddings (nct_id, model, variant, text_hash, embedding) VALUES ('A', %s, 'with_inclusion', 'h', '[0,1,0]')", (MODEL,))
    assert ids(run(conn, mode="vector")) == ["A"]
    assert ids(run(conn, mode="vector", variant=WITH_INCLUSION)) == ["A"]


def test_missing_embeddings_gives_a_helpful_error(conn):
    add(conn, "A", "tumor", None)
    with pytest.raises(SearchError, match="embed_trials --model fake/model"):
        run(conn)
    assert ids(run(conn, mode="fts")) == ["A"]  # text search needs no embeddings


def test_wrong_size_query_vector_gives_a_helpful_error(conn):
    add(conn, "A", "tumor", X)
    with pytest.raises(SearchError, match="does not match"):
        run(conn, encoder=QueryEncoder([1.0, 0.0]))


def test_query_prefix_is_sent_for_bge_models_only(conn):
    add(conn, "A", "tumor", X, model="BAAI/bge-small-en-v1.5")
    enc = QueryEncoder()
    run(conn, query="lung cancer", encoder=enc, model="BAAI/bge-small-en-v1.5")
    assert enc.texts == ["Represent this sentence for searching relevant passages: lung cancer"]
    add(conn, "B", "tumor", X, model="x/minilm")
    enc2 = QueryEncoder()
    run(conn, query="lung cancer", encoder=enc2, model="x/minilm")
    assert enc2.texts == ["lung cancer"]


def test_fts_mode_needs_no_encoder_but_vector_modes_do(conn):
    add(conn, "A", "tumor", X)
    assert ids(search(conn, "tumor", age=50, sex="FEMALE", mode="fts")) == ["A"]
    with pytest.raises(ValueError, match="encoder"):
        search(conn, "tumor", age=50, sex="FEMALE", mode="hybrid", model=MODEL)


@pytest.mark.parametrize("kwargs", [
    {"sex": "OTHER"}, {"age": -1}, {"age": 200}, {"k": 0}, {"pool": 0}, {"mode": "magic"}, {"variant": "x"},
])
def test_bad_inputs_are_rejected(conn, kwargs):
    with pytest.raises(ValueError):
        run(conn, **kwargs)


def test_query_text_is_never_logged(conn, caplog):
    add(conn, "A", "tumor", X)
    with caplog.at_level(logging.DEBUG):
        run(conn, query="SECRET-PROFILE-TEXT tumor")
    assert "SECRET-PROFILE-TEXT" not in caplog.text


# --- command line ---------------------------------------------------------------------


@pytest.fixture
def cli(conn, pg_url, monkeypatch, capsys):
    import psycopg

    schema = conn.execute("SELECT current_schema()").fetchone()[0]
    monkeypatch.setattr(search_module, "connect", lambda url: psycopg.connect(
        url, autocommit=True, options=f"-c search_path={schema},public"))
    monkeypatch.setattr(search_module, "SentenceTransformerEncoder", lambda name: QueryEncoder())

    def run_cli(*args):
        code = search_module.main([*args, "--database-url", pg_url])
        return code, capsys.readouterr()

    return run_cli


def test_cli_prints_top_10_titles_with_the_disclaimer(conn, cli):
    for i in range(15):
        add(conn, f"NCT{i:02d}", f"Tumor study number {i}", [1.0, i * 0.01, 0.0], model="BAAI/bge-small-en-v1.5")
    code, out = cli("tumor", "--age", "58", "--sex", "female")
    assert code == 0
    assert out.out.count("Tumor study number") == 10
    assert " 1. NCT00" in out.out and "NCT09" in out.out and "NCT10" not in out.out
    assert out.out.rstrip().endswith(DISCLAIMER)


def test_cli_show_and_mode_options(conn, cli):
    for i in range(4):
        add(conn, f"NCT{i}", f"Tumor study {i}", X, model="BAAI/bge-small-en-v1.5")
    code, out = cli("tumor", "--age", "40", "--sex", "MALE", "--show", "2", "--mode", "fts")
    assert code == 0 and out.out.count("Tumor study") == 2 and DISCLAIMER in out.out


def test_cli_no_results_still_carries_the_disclaimer(conn, cli):
    add(conn, "NCT1", "tumor", X, model="BAAI/bge-small-en-v1.5")
    code, out = cli("zebra", "--age", "40", "--sex", "MALE", "--mode", "fts")
    assert code == 0 and "No matching trials." in out.out and DISCLAIMER in out.out


def test_cli_reports_missing_embeddings_on_stderr(conn, cli):
    add(conn, "NCT1", "tumor", None)
    code, out = cli("tumor", "--age", "40", "--sex", "MALE")
    assert code == 1 and "embed_trials" in out.err


@pytest.mark.parametrize("args", [
    ["q", "--sex", "FEMALE"], ["q", "--age", "40"], ["q", "--age", "40", "--sex", "X"],
    ["q", "--age", "999", "--sex", "MALE"], ["q", "--age", "40", "--sex", "MALE", "--show", "0"],
])
def test_cli_rejects_bad_arguments(args):
    with pytest.raises(SystemExit):
        search_module.main([*args, "--database-url", "postgresql://x"])

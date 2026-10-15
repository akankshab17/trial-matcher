import csv
import json
import math

import pytest

from tests.test_search import X, Y, Z, QueryEncoder, add
from trial_matcher import evaluate
from trial_matcher.evaluate import (
    LABEL_COLUMNS,
    LabelRow,
    Profile,
    criteria_excerpt,
    extend_labels,
    format_report,
    format_status,
    grades_by_profile,
    hit_at_k,
    load_labels,
    load_profiles,
    ndcg_at_k,
    recall_at_k,
    reviewed_profile_ids,
    run_eval,
    save_labels,
    score_profile,
    summarize_method,
)
from trial_matcher.search import DISCLAIMER

MODEL = "fake/model"


# --- metrics -----------------------------------------------------------------------------


def test_recall_at_k():
    assert recall_at_k(["a", "x", "b"], {"a", "b", "c"}, 50) == pytest.approx(2 / 3)
    assert recall_at_k(["a", "x", "b"], {"a", "b", "c"}, 1) == pytest.approx(1 / 3)
    assert recall_at_k(["x"], {"a"}, 50) == 0.0


def test_hit_at_k():
    assert hit_at_k(["x", "a"], {"a"}, 10) == 1.0
    assert hit_at_k(["x", "y", "a"], {"a"}, 2) == 0.0


def test_ndcg_known_value():
    grades = {"a": 2, "b": 1}
    got = ndcg_at_k(["b", "a"], grades, 10)
    dcg = 1 / math.log2(2) + 2 / math.log2(3)
    ideal = 2 / math.log2(2) + 1 / math.log2(3)
    assert got == pytest.approx(dcg / ideal)


def test_ndcg_perfect_zero_and_unlabelled():
    grades = {"a": 2, "b": 1, "c": 0}
    assert ndcg_at_k(["a", "b", "c"], grades, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["c", "zzz"], grades, 10) == 0.0  # unlabelled counts as 0
    assert ndcg_at_k(["x", "y"], {"a": 0}, 10) == 0.0  # nothing relevant: no division by zero
    assert ndcg_at_k(["x", "a"], {"a": 2}, 1) == 0.0  # a is cut off by k


def test_score_profile_respects_min_grade_and_skips_profiles_with_nothing_to_find():
    grades = {"a": 1, "b": 2}
    strict = score_profile(["a", "b"], grades, 2)
    lenient = score_profile(["a", "b"], grades, 1)
    assert strict.recall == 1.0 and lenient.recall == 1.0
    assert score_profile(["a"], grades, 2).recall == 0.0
    assert score_profile(["a"], grades, 1).recall == 0.5
    assert score_profile(["a"], {"a": 1}, 2) is None


def test_judged_at_10_counts_labelled_results():
    s = score_profile(["a", "unlabelled", "b", "c"], {"a": 2, "b": 0}, 2)
    assert s.judged_at_10 == pytest.approx(2 / 4)


def test_summarize_breaks_out_only_tags_with_enough_profiles():
    scores = {p: score_profile(["a"], {"a": 2}, 2) for p in ("P1", "P2", "P3", "P4")}
    tags = {"P1": ("big", "tiny"), "P2": ("big",), "P3": ("big",), "P4": ()}
    rep = summarize_method("hybrid", scores, tags)
    assert rep.profiles_scored == 4 and set(rep.by_tag) == {"big"} and rep.by_tag["big"].profiles_scored == 3


# --- profiles and labels ---------------------------------------------------------------------


def write_profiles(tmp_path, items):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(items))
    return path


GOOD = {"id": "P1", "text": "lung cancer", "age": 50, "sex": "female", "tags": ["x"]}


def test_load_profiles_normalises_and_validates(tmp_path):
    [p] = load_profiles(write_profiles(tmp_path, [GOOD]))
    assert (p.id, p.age, p.sex, p.tags) == ("P1", 50.0, "FEMALE", ("x",))
    for bad in ({**GOOD, "sex": "other"}, {**GOOD, "age": -2}, {**GOOD, "age": 500}, {**GOOD, "text": " "}):
        with pytest.raises(ValueError):
            load_profiles(write_profiles(tmp_path, [bad]))
    with pytest.raises(ValueError, match="duplicate"):
        load_profiles(write_profiles(tmp_path, [GOOD, GOOD]))


def test_labels_round_trip_with_awkward_text(tmp_path):
    rows = [
        LabelRow("P1", "NCT1", grade=2, suggested_grade=1, reviewed=True, reason='has "quotes", commas',
                 profile="line one\nline two", title="Título ≥ 18", conditions="a; b",
                 key_criteria="I: x | E: y", url="https://x"),
        LabelRow("P1", "NCT2"),
    ]
    path = tmp_path / "labels.csv"
    save_labels(rows, path)
    assert load_labels(path) == rows
    assert next(csv.reader(open(path, encoding="utf-8"))) == LABEL_COLUMNS


def test_load_labels_accepts_an_excel_bom_and_yes(tmp_path):
    path = tmp_path / "labels.csv"
    save_labels([LabelRow("P1", "NCT1", grade=1, suggested_grade=1, reviewed=True)], path)
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes().replace(b",y,", b",YES,"))
    [row] = load_labels(path)
    assert row.reviewed and row.grade == 1


def test_load_labels_rejects_bad_files(tmp_path):
    path = tmp_path / "labels.csv"

    def write(**over):
        r = {c: "" for c in LABEL_COLUMNS} | {"profile_id": "P1", "nct_id": "NCT1"} | over
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, LABEL_COLUMNS)
            w.writeheader()
            w.writerow(r)
            return w

    for bad in ({"grade": "3"}, {"grade": "two"}, {"suggested_grade": "9"}, {"reviewed": "maybe"}):
        write(**bad)
        with pytest.raises(ValueError):
            load_labels(path)
    write()
    with pytest.raises(ValueError, match="unknown profiles"):
        load_labels(path, profile_ids=["P2"])
    assert len(load_labels(path, profile_ids=["P1"])) == 1

    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(["P1", "NCT1"] + [""] * (len(LABEL_COLUMNS) - 2))
    with pytest.raises(ValueError, match="duplicate"):
        load_labels(path)

    path.write_text("profile_id,nct_id\nP1,NCT1\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_labels(path)


def test_reviewed_profiles_need_every_row_reviewed():
    rows = [LabelRow("P1", "A", reviewed=True), LabelRow("P1", "B", reviewed=True),
            LabelRow("P2", "A", reviewed=True), LabelRow("P2", "B")]
    assert reviewed_profile_ids(rows) == {"P1"}


def test_grades_by_profile_skips_blank_grades():
    rows = [LabelRow("P1", "A", grade=2), LabelRow("P1", "B"), LabelRow("P2", "A", grade=0)]
    assert grades_by_profile(rows) == {"P1": {"A": 2}, "P2": {"A": 0}}


# --- criteria excerpt -------------------------------------------------------------------------


def test_excerpt_shows_first_inclusions_then_treatment_related_criteria():
    crit = [("inclusion", "Age 18+"), ("inclusion", "ECOG 0-1"), ("inclusion", "Measurable disease"),
            ("inclusion", "Adequate liver function"), ("inclusion", "Progressed after first-line gemcitabine"),
            ("exclusion", "Pregnant"), ("exclusion", "Prior treatment with drug X"), ("unknown", "Previous therapy note")]
    out = criteria_excerpt(crit)
    assert out.startswith("I: Age 18+ | I: ECOG 0-1 | I: Measurable disease")
    assert "I: Progressed after first-line gemcitabine" in out
    assert "E: Prior treatment with drug X" in out
    assert "Pregnant" not in out and "Adequate liver" not in out and "Previous therapy note" not in out


def test_excerpt_truncates_and_handles_empty():
    assert criteria_excerpt([]) == ""
    out = criteria_excerpt([("inclusion", "word " * 200)] * 3)
    assert all(len(part) <= 164 for part in out.split(" | ")) and len(out) <= 1000
    assert criteria_excerpt([("inclusion", "multi\n- line   text")]) == "I: multi - line text"


# --- pooling and running on a real throwaway Postgres ---------------------------------------------


def make_world(conn):
    """12 'near' distractors beat target A in vector space; only A shares words with the query."""
    add(conn, "A", "lung tumor trial", [1.0, 0.9, 0.0], model=MODEL)
    for i in range(12):
        add(conn, f"D{i:02d}", f"unrelated study {i}", [1.0, 0.01 * i, 0.0], model=MODEL)
    add(conn, "Z", "lung tumor other", Z, model=MODEL)


PROFILE = Profile(id="P1", text="lung tumor", age=50, sex="FEMALE", tags=("t",))


def test_pool_adds_top_n_of_every_method_with_details(conn):
    make_world(conn)
    rows, added = extend_labels(conn, [PROFILE], [], QueryEncoder(X), MODEL, "basic", depth=3)
    ids = {r.nct_id for r in rows}
    assert added == len(rows) == len(ids)
    assert {"D00", "D01", "D02"} <= ids  # top 3 by vector
    assert {"A", "Z"} <= ids  # the only full-text matches
    a = next(r for r in rows if r.nct_id == "A")
    assert (a.profile_id, a.title, a.url, a.profile) == ("P1", "lung tumor trial", "https://clinicaltrials.gov/study/A", "lung tumor")
    assert a.grade is None and a.suggested_grade is None and not a.reviewed


def test_pool_never_touches_existing_rows_and_is_idempotent(conn):
    make_world(conn)
    kept = LabelRow("P1", "A", grade=2, suggested_grade=1, reviewed=True, reason="human edit", title="old title")
    rows, added = extend_labels(conn, [PROFILE], [kept], QueryEncoder(X), MODEL, "basic", depth=3)
    assert kept in rows and sum(r.nct_id == "A" for r in rows) == 1
    again, added_again = extend_labels(conn, [PROFILE], rows, QueryEncoder(X), MODEL, "basic", depth=3)
    assert added_again == 0 and again == rows
    deeper, added_deeper = extend_labels(conn, [PROFILE], rows, QueryEncoder(X), MODEL, "basic", depth=6)
    assert added_deeper > 0 and all(r in deeper for r in rows)


def graded(**g):
    return [LabelRow("P1", n, grade=v, suggested_grade=v) for n, v in g.items()]


def test_run_eval_separates_the_methods(conn):
    make_world(conn)
    labels = graded(A=2, Z=1, D00=0, D01=0)
    rep = run_eval(conn, [PROFILE], labels, QueryEncoder(X), MODEL)
    by = {m.method: m for m in rep.methods}
    assert by["fts"].hit_at_10 == 1.0 and by["fts"].recall_at_50 == 1.0
    assert by["vector"].hit_at_10 == 0.0 and by["vector"].recall_at_50 == 1.0  # A is 13th by vector
    assert by["hybrid"].hit_at_10 == 1.0  # in both lists, so it beats the vector-only leaders
    assert by["fts"].ndcg_at_10 > by["vector"].ndcg_at_10
    assert (rep.profiles_used, rep.profiles_total) == (1, 1)


def test_run_eval_min_grade_and_judged_coverage(conn):
    make_world(conn)
    labels = graded(A=1, Z=1)
    strict = run_eval(conn, [PROFILE], labels, QueryEncoder(X), MODEL, methods=["vector"], min_grade=2)
    assert strict.profiles_unscorable == ["P1"] and strict.methods[0].profiles_scored == 0
    lenient = run_eval(conn, [PROFILE], labels, QueryEncoder(X), MODEL, methods=["vector"], min_grade=1)
    assert lenient.methods[0].profiles_scored == 1
    assert lenient.methods[0].judged_at_10 < 0.5  # most of vector's top 10 (the D's) are unlabelled


def test_run_eval_only_reviewed_and_unlabelled_profiles(conn):
    make_world(conn)
    p2 = Profile(id="P2", text="lung tumor", age=50, sex="FEMALE")
    labels = graded(A=2) + [LabelRow("P2", "A", grade=2, suggested_grade=2, reviewed=True)]
    both = run_eval(conn, [PROFILE, p2, Profile("P3", "x", 40, "MALE")], labels, None, MODEL, methods=["fts"])
    assert both.profiles_used == 2  # P3 has no labels
    only = run_eval(conn, [PROFILE, p2], labels, None, MODEL, methods=["fts"], only_reviewed=True)
    assert only.profiles_used == 1 and only.profiles_reviewed == 1


def test_report_text_has_the_table_tags_and_disclaimer(conn):
    make_world(conn)
    rep = run_eval(conn, [PROFILE], graded(A=2), QueryEncoder(X), MODEL, methods=["fts", "vector"])
    text = format_report(rep)
    assert "recall@50" in text and "nDCG@10" in text and "hit@10" in text
    assert "| fts " in text and "| vector " in text
    assert text.rstrip().endswith(DISCLAIMER)
    assert "Relevant = grade >= 2" in text


def test_status_reports_progress_and_changes():
    profiles = [Profile("P1", "a", 1, "MALE"), Profile("P2", "b", 1, "MALE"), Profile("P3", "c", 1, "MALE")]
    rows = [LabelRow("P1", "A", grade=2, suggested_grade=2, reviewed=True),
            LabelRow("P1", "B", grade=0, suggested_grade=1, reviewed=True),
            LabelRow("P2", "A", grade=1, suggested_grade=1)]
    text = format_status(rows, profiles)
    assert "Profiles reviewed: 1 of 3" in text
    assert "changed by the reviewer: 1 of 2 (50%)" in text
    assert "1->0: 1" in text and "2->2: 1" in text
    assert "no rows" in text and "TO REVIEW" in text


# --- command line ---------------------------------------------------------------------------------


@pytest.fixture
def cli(conn, pg_url, monkeypatch, tmp_path, capsys):
    import psycopg

    schema = conn.execute("SELECT current_schema()").fetchone()[0]
    monkeypatch.setattr(evaluate, "connect", lambda url: psycopg.connect(
        url, autocommit=True, options=f"-c search_path={schema},public"))
    monkeypatch.setattr(evaluate, "SentenceTransformerEncoder", lambda name: QueryEncoder(X))
    profiles = write_profiles(tmp_path, [{"id": "P1", "text": "lung tumor", "age": 50, "sex": "FEMALE", "tags": []}])
    labels = tmp_path / "labels.csv"

    def run(*args):
        code = evaluate.main([*args, "--profiles", str(profiles), "--labels", str(labels)]
                             + (["--database-url", pg_url] if args[0] != "status" else []))
        captured = capsys.readouterr()
        run.err = captured.err
        return code, captured.out

    run.labels = labels
    return run


def test_cli_pool_then_run(conn, cli, tmp_path):
    add(conn, "A", "lung tumor trial", X, model="BAAI/bge-small-en-v1.5")
    code, out = cli("pool", "--depth", "5")
    assert code == 0 and "added 1 new rows" in out
    rows = load_labels(cli.labels)
    save_labels([LabelRow(**{**r.__dict__, "grade": 2}) for r in rows], cli.labels)
    report_path = tmp_path / "out" / "report.md"
    code, out = cli("run", "--out", str(report_path))
    assert code == 0 and "hybrid" in out and DISCLAIMER in out
    assert report_path.read_text().strip() == out.strip()
    code, out = cli("status")
    assert code == 0 and "P1" in out


def test_cli_run_without_labels_explains_what_to_do(cli):
    assert cli("run")[0] == 1
    assert "pool" in cli.err


def test_cli_run_fts_only_needs_no_model(conn, cli, monkeypatch):
    add(conn, "A", "lung tumor trial", X)
    save_labels([LabelRow("P1", "A", grade=2, suggested_grade=2)], cli.labels)
    monkeypatch.setattr(evaluate, "SentenceTransformerEncoder", lambda name: pytest.fail("model loaded"))
    code, out = cli("run", "--methods", "fts")
    assert code == 0 and "| fts " in out and "| vector " not in out


# --- the committed eval data -------------------------------------------------------------------------


def test_committed_profiles_are_valid_and_varied():
    profiles = load_profiles()
    assert len(profiles) == 30
    assert sum("prior_treatment" in p.tags for p in profiles) >= 10  # the nuance the eval must measure
    assert {p.sex for p in profiles} == {"FEMALE", "MALE"}
    assert min(p.age for p in profiles) < 18 < max(p.age for p in profiles) > 70
    assert all(p.relevant_if and p.not_relevant_if for p in profiles)


def test_committed_labels_are_complete_and_consistent():
    profiles = load_profiles()
    rows = load_labels(profile_ids=[p.id for p in profiles])
    assert {r.profile_id for r in rows} == {p.id for p in profiles}  # every profile has candidates
    for r in rows:
        assert r.grade in (0, 1, 2) and r.suggested_grade in (0, 1, 2), (r.profile_id, r.nct_id)
        assert r.title and r.url.endswith(r.nct_id)
        if not r.reviewed:
            assert r.grade == r.suggested_grade  # untouched suggestions stay identical until reviewed
    graded = {pid: [r.grade for r in rows if r.profile_id == pid] for pid in {r.profile_id for r in rows}}
    assert sum(any(g == 2 for g in gs) for gs in graded.values()) >= 25  # enough profiles have something to find

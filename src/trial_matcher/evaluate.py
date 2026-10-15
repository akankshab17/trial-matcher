"""Retrieval eval: pool candidates, review labels, measure recall@50, nDCG@10 and hit@10.

Files in /evals (see evals/README.md):
    profiles.json   synthetic patient profiles (no real patient data)
    labels.csv      graded trials per profile; a human reviews and edits the `grade` column

Commands (all run from the repo root):
    python -m trial_matcher.evaluate pool     add new candidates (top 10 per method) to labels.csv
    python -m trial_matcher.evaluate status   review progress and how often the suggestions were changed
    python -m trial_matcher.evaluate run      score fts / vector / hybrid and print a report

Grades: 2 = right disease and nothing stated excludes the patient; 1 = right disease but fit is
uncertain; 0 = wrong disease, or a stated requirement clearly excludes the patient.
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import psycopg

from trial_matcher import config
from trial_matcher.db import connect
from trial_matcher.embed import BASIC, DEFAULT_MODEL, VARIANTS, Encoder, SentenceTransformerEncoder
from trial_matcher.search import DISCLAIMER, SEXES, search

EVALS_DIR = Path(__file__).resolve().parents[2] / "evals"
PROFILES_PATH = EVALS_DIR / "profiles.json"
LABELS_PATH = EVALS_DIR / "labels.csv"

METHODS = ("fts", "vector", "hybrid")
GRADES = (0, 1, 2)
POOL_DEPTH = 10  # candidates taken from the top of each method into the label pool
K_RECALL = 50
K_RANK = 10
MIN_TAG_PROFILES = 3  # tags with fewer profiles are not broken out

# Order matters: it is the column order a reviewer sees. `suggested_grade` stays last and untouched.
LABEL_COLUMNS = [
    "profile_id", "nct_id", "grade", "reviewed", "reason",
    "profile", "title", "conditions", "key_criteria", "url", "suggested_grade",
]


# --- profiles ------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    id: str
    text: str
    age: float
    sex: str
    tags: tuple[str, ...] = ()
    relevant_if: str = ""
    not_relevant_if: str = ""


def load_profiles(path: Path = PROFILES_PATH) -> list[Profile]:
    profiles = []
    for raw in json.loads(Path(path).read_text()):
        p = Profile(
            id=raw["id"], text=raw["text"], age=float(raw["age"]), sex=raw["sex"].upper(),
            tags=tuple(raw.get("tags", [])), relevant_if=raw.get("relevant_if", ""),
            not_relevant_if=raw.get("not_relevant_if", ""),
        )
        if not p.id or not p.text.strip():
            raise ValueError(f"profile needs an id and text: {raw}")
        if p.sex not in SEXES:
            raise ValueError(f"{p.id}: sex must be one of {SEXES}, got {p.sex!r}")
        if not 0 <= p.age <= 130:
            raise ValueError(f"{p.id}: age out of range: {p.age}")
        profiles.append(p)
    ids = [p.id for p in profiles]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate profile ids: {[i for i, n in Counter(ids).items() if n > 1]}")
    return profiles


# --- labels --------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelRow:
    profile_id: str
    nct_id: str
    grade: int | None = None  # None = not graded yet
    suggested_grade: int | None = None  # the first-pass suggestion; never edited by the reviewer
    reviewed: bool = False
    reason: str = ""
    profile: str = ""  # display only, for the reviewer
    title: str = ""
    conditions: str = ""
    key_criteria: str = ""
    url: str = ""


def _parse_grade(value: str, where: str) -> int | None:
    value = (value or "").strip()
    if value == "":
        return None
    if value not in {str(g) for g in GRADES}:
        raise ValueError(f"{where}: grade must be 0, 1 or 2 (or blank), got {value!r}")
    return int(value)


def load_labels(path: Path = LABELS_PATH, profile_ids: Iterable[str] | None = None) -> list[LabelRow]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in LABEL_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        rows, seen = [], set()
        for n, r in enumerate(reader, start=2):
            where = f"{path}:{n}"
            key = (r["profile_id"].strip(), r["nct_id"].strip())
            if key in seen:
                raise ValueError(f"{where}: duplicate row for {key}")
            seen.add(key)
            reviewed = (r["reviewed"] or "").strip().lower()
            if reviewed not in ("", "y", "yes", "n", "no"):
                raise ValueError(f"{where}: reviewed must be y or blank, got {r['reviewed']!r}")
            rows.append(LabelRow(
                profile_id=key[0], nct_id=key[1], grade=_parse_grade(r["grade"], where),
                suggested_grade=_parse_grade(r["suggested_grade"], where),
                reviewed=reviewed in ("y", "yes"), reason=r["reason"], profile=r["profile"], title=r["title"],
                conditions=r["conditions"], key_criteria=r["key_criteria"], url=r["url"],
            ))
    if profile_ids is not None:
        known = set(profile_ids)
        unknown = sorted({r.profile_id for r in rows} - known)
        if unknown:
            raise ValueError(f"{path}: labels refer to unknown profiles {unknown}")
    return rows


def save_labels(rows: Sequence[LabelRow], path: Path = LABELS_PATH) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({
                "profile_id": r.profile_id, "nct_id": r.nct_id,
                "grade": "" if r.grade is None else r.grade, "reviewed": "y" if r.reviewed else "",
                "reason": r.reason, "profile": r.profile, "title": r.title, "conditions": r.conditions,
                "key_criteria": r.key_criteria, "url": r.url,
                "suggested_grade": "" if r.suggested_grade is None else r.suggested_grade,
            })


def reviewed_profile_ids(rows: Sequence[LabelRow]) -> set[str]:
    """A profile counts as reviewed when it has rows and every one is marked reviewed."""
    by_profile: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_profile[r.profile_id].append(r.reviewed)
    return {pid for pid, flags in by_profile.items() if all(flags)}


def grades_by_profile(rows: Sequence[LabelRow]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for r in rows:
        if r.grade is not None:
            out[r.profile_id][r.nct_id] = r.grade
    return out


# --- what the reviewer sees ------------------------------------------------------------

_TREATMENT_WORDS = re.compile(
    r"prior|previous|refractory|progress|relaps|line\b|naive|na[iï]ve|untreated|resistan|failed|received|"
    r"treated|therapy|pretreat|recurren|adjuvant|metasta|stage", re.IGNORECASE)


def criteria_excerpt(criteria: Sequence[tuple[str, str]], head: int = 3, extra: int = 4,
                     item_chars: int = 160, total_chars: int = 1000) -> str:
    """A short, readable summary of a trial's criteria for a human grader.

    The first few inclusion criteria, then up to `extra` more criteria (inclusion or exclusion)
    that mention prior treatment, line of therapy, stage or setting. I: = inclusion, E: = exclusion.
    """
    def short(text: str) -> str:
        text = " ".join(text.split())
        return text if len(text) <= item_chars else text[: item_chars - 1] + "…"

    parts, used = [], set()
    inclusion = [(i, t) for i, (k, t) in enumerate(criteria) if k == "inclusion"]
    for i, text in inclusion[:head]:
        parts.append(f"I: {short(text)}")
        used.add(i)
    for i, (kind, text) in enumerate(criteria):
        if len(used) >= head + extra:
            break
        if i not in used and kind in ("inclusion", "exclusion") and _TREATMENT_WORDS.search(text):
            parts.append(f"{'I' if kind == 'inclusion' else 'E'}: {short(text)}")
            used.add(i)
    return " | ".join(parts)[:total_chars]


def trial_url(nct_id: str) -> str:
    return f"https://clinicaltrials.gov/study/{nct_id}"


# --- metrics ----------------------------------------------------------------------------


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def ndcg_at_k(ranked: Sequence[str], grades: dict[str, int], k: int) -> float:
    """Gain = grade (0/1/2), discount 1/log2(rank+1); unlabelled results count as 0."""
    dcg = sum(grades.get(n, 0) / math.log2(i + 1) for i, n in enumerate(ranked[:k], start=1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class ProfileScores:
    recall: float
    ndcg: float
    hit: float
    judged_at_10: float  # share of the top 10 that has any label (low = the label pool misses this method)


def score_profile(ranked: Sequence[str], grades: dict[str, int], min_grade: int) -> ProfileScores | None:
    """None when the profile has no relevant trial (nothing to find, so it cannot be scored)."""
    relevant = {n for n, g in grades.items() if g >= min_grade}
    if not relevant:
        return None
    top = ranked[:K_RANK]
    return ProfileScores(
        recall=recall_at_k(ranked, relevant, K_RECALL),
        ndcg=ndcg_at_k(ranked, grades, K_RANK),
        hit=hit_at_k(ranked, relevant, K_RANK),
        judged_at_10=(sum(1 for n in top if n in grades) / len(top)) if top else 1.0,
    )


@dataclass
class MethodReport:
    method: str
    profiles_scored: int
    recall_at_50: float
    ndcg_at_10: float
    hit_at_10: float
    judged_at_10: float
    by_tag: dict[str, "MethodReport"] = field(default_factory=dict)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_method(method: str, scores: dict[str, ProfileScores], tags: dict[str, tuple[str, ...]]) -> MethodReport:
    def build(subset: dict[str, ProfileScores]) -> MethodReport:
        vals = list(subset.values())
        return MethodReport(method, len(vals), _mean([s.recall for s in vals]), _mean([s.ndcg for s in vals]),
                            _mean([s.hit for s in vals]), _mean([s.judged_at_10 for s in vals]))

    report = build(scores)
    tag_counts = Counter(t for pid in scores for t in tags.get(pid, ()))
    for tag, n in sorted(tag_counts.items()):
        if n >= MIN_TAG_PROFILES:
            report.by_tag[tag] = build({pid: s for pid, s in scores.items() if tag in tags.get(pid, ())})
    return report


# --- pooling ----------------------------------------------------------------------------


def run_method(conn: psycopg.Connection, profile: Profile, method: str, encoder: Encoder | None,
               model: str, variant: str, k: int = K_RECALL) -> list[str]:
    results = search(conn, profile.text, age=profile.age, sex=profile.sex, encoder=encoder,
                     model=model, variant=variant, k=k, mode=method)
    return [r.nct_id for r in results]


def fetch_trial_info(conn: psycopg.Connection, nct_ids: Sequence[str]) -> dict[str, dict]:
    info = {}
    for nct, title, conditions, kinds, texts in conn.execute(
        """SELECT t.nct_id, t.title, t.conditions,
                  (SELECT array_agg(c.kind ORDER BY c.position) FROM criteria c WHERE c.nct_id = t.nct_id),
                  (SELECT array_agg(c.text ORDER BY c.position) FROM criteria c WHERE c.nct_id = t.nct_id)
           FROM trials t WHERE t.nct_id = ANY(%s)""", (list(nct_ids),)):
        info[nct] = {"title": title, "conditions": "; ".join(conditions),
                     "key_criteria": criteria_excerpt(list(zip(kinds or [], texts or [])))}
    return info


def extend_labels(conn: psycopg.Connection, profiles: Sequence[Profile], existing: Sequence[LabelRow],
                  encoder: Encoder | None, model: str, variant: str, depth: int = POOL_DEPTH,
                  methods: Sequence[str] = METHODS) -> tuple[list[LabelRow], int]:
    """Add the top `depth` of every method as ungraded rows. Existing rows are never touched."""
    have = {(r.profile_id, r.nct_id) for r in existing}
    wanted: list[tuple[Profile, str]] = []
    for p in profiles:
        for m in methods:
            for nct in run_method(conn, p, m, encoder, model, variant, k=depth)[:depth]:
                if (p.id, nct) not in have:
                    have.add((p.id, nct))
                    wanted.append((p, nct))
    info = fetch_trial_info(conn, sorted({n for _, n in wanted}))
    new_rows = [
        LabelRow(profile_id=p.id, nct_id=n, profile=p.text, url=trial_url(n), **info[n])
        for p, n in wanted if n in info
    ]
    order = {p.id: i for i, p in enumerate(profiles)}
    merged = sorted([*existing, *new_rows], key=lambda r: (order.get(r.profile_id, 10**6), r.nct_id))
    return merged, len(new_rows)


# --- running the eval ----------------------------------------------------------------------


@dataclass
class EvalReport:
    methods: list[MethodReport]
    profiles_total: int
    profiles_reviewed: int
    profiles_used: int
    profiles_unscorable: list[str]
    min_grade: int
    model: str
    variant: str


def run_eval(conn: psycopg.Connection, profiles: Sequence[Profile], labels: Sequence[LabelRow],
             encoder: Encoder | None, model: str = DEFAULT_MODEL, variant: str = BASIC,
             methods: Sequence[str] = METHODS, min_grade: int = 2, only_reviewed: bool = False) -> EvalReport:
    reviewed = reviewed_profile_ids(labels)
    grades = grades_by_profile(labels)
    used = [p for p in profiles if p.id in grades and (not only_reviewed or p.id in reviewed)]
    tags = {p.id: p.tags for p in profiles}
    reports, unscorable = [], sorted(p.id for p in used if not any(g >= min_grade for g in grades[p.id].values()))
    for m in methods:
        scores = {}
        for p in used:
            s = score_profile(run_method(conn, p, m, encoder, model, variant), grades[p.id], min_grade)
            if s is not None:
                scores[p.id] = s
        reports.append(summarize_method(m, scores, tags))
    return EvalReport(reports, len(profiles), len(reviewed & {p.id for p in profiles}), len(used),
                      unscorable, min_grade, model, variant)


def format_report(report: EvalReport) -> str:
    def row(r: MethodReport, label: str | None = None) -> str:
        return (f"| {label or r.method:<24} | {r.profiles_scored:>3} | {r.recall_at_50:>9.3f} | {r.ndcg_at_10:>7.3f} "
                f"| {r.hit_at_10:>6.3f} | {r.judged_at_10:>9.2f} |")

    head = ("| method / group           |   n | recall@50 | nDCG@10 | hit@10 | judged@10 |\n"
            "|--------------------------|-----|-----------|---------|--------|-----------|")
    lines = [
        f"Retrieval eval: model {report.model}, text variant {report.variant}",
        f"Relevant = grade >= {report.min_grade}. Profiles scored: {report.profiles_used} of {report.profiles_total} "
        f"({report.profiles_reviewed} reviewed by a human).",
    ]
    if report.profiles_unscorable:
        lines.append(f"No relevant trial in the labels (skipped): {', '.join(report.profiles_unscorable)}")
    lines += ["", "All profiles", head, *(row(m) for m in report.methods)]
    tags = sorted({t for m in report.methods for t in m.by_tag})
    for tag in tags:
        lines += ["", f"Tag: {tag}", head, *(row(m.by_tag[tag], m.method) for m in report.methods if tag in m.by_tag)]
    lines += ["", "n = profiles scored. judged@10 = share of a method's top 10 that has a label (1.00 = fully judged).",
              "", DISCLAIMER]
    return "\n".join(lines)


def format_status(rows: Sequence[LabelRow], profiles: Sequence[Profile]) -> str:
    by_profile: dict[str, list[LabelRow]] = defaultdict(list)
    for r in rows:
        by_profile[r.profile_id].append(r)
    reviewed = reviewed_profile_ids(rows)
    lines = [f"{'profile':8} {'rows':>5} {'graded':>7} {'reviewed':>9} {'changed':>8}  status"]
    changed_total = compared = 0
    confusion: Counter = Counter()
    for p in profiles:
        rs = by_profile.get(p.id, [])
        graded = sum(r.grade is not None for r in rs)
        rev = sum(r.reviewed for r in rs)
        ch = [r for r in rs if r.reviewed and r.suggested_grade is not None and r.grade != r.suggested_grade]
        done = [r for r in rs if r.reviewed and r.suggested_grade is not None]
        changed_total, compared = changed_total + len(ch), compared + len(done)
        for r in done:
            confusion[(r.suggested_grade, r.grade)] += 1
        lines.append(f"{p.id:8} {len(rs):5d} {graded:7d} {rev:9d} {len(ch):8d}  "
                     f"{'reviewed' if p.id in reviewed else 'TO REVIEW' if rs else 'no rows (run pool)'}")
    lines.append("")
    lines.append(f"Profiles reviewed: {len(reviewed & {p.id for p in profiles})} of {len(profiles)}")
    if compared:
        lines.append(f"Labels changed by the reviewer: {changed_total} of {compared} "
                     f"({100 * changed_total / compared:.0f}%)")
        lines.append("Suggested -> reviewed grade counts: " +
                     ", ".join(f"{s}->{g}: {n}" for (s, g), n in sorted(confusion.items())))
    return "\n".join(lines)


# --- command line -------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval eval for trial search.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, helptext in (("pool", "add top-10-per-method candidates to labels.csv"),
                           ("status", "review progress"), ("run", "score fts / vector / hybrid")):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--profiles", type=Path, default=PROFILES_PATH)
        s.add_argument("--labels", type=Path, default=LABELS_PATH)
        if name != "status":
            s.add_argument("--model", default=DEFAULT_MODEL)
            s.add_argument("--variant", choices=VARIANTS, default=BASIC)
            s.add_argument("--database-url", default=None, help="default: $DATABASE_URL")
        if name == "pool":
            s.add_argument("--depth", type=int, default=POOL_DEPTH, help="top N per method (default %(default)s)")
        if name == "run":
            s.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
            s.add_argument("--min-grade", type=int, choices=(1, 2), default=2,
                           help="grade that counts as relevant (default %(default)s)")
            s.add_argument("--only-reviewed", action="store_true", help="score only fully reviewed profiles")
            s.add_argument("--out", type=Path, default=None, help="also write the report to this file")
    args = parser.parse_args(argv)

    profiles = load_profiles(args.profiles)
    if args.command == "status":
        print(format_status(load_labels(args.labels, [p.id for p in profiles]), profiles))
        return 0

    existing = load_labels(args.labels, [p.id for p in profiles]) if args.labels.exists() else []
    needs_vectors = args.command == "pool" or any(m != "fts" for m in args.methods)
    encoder = SentenceTransformerEncoder(args.model) if needs_vectors else None
    with connect(args.database_url or config.database_url()) as conn:
        if args.command == "pool":
            if args.depth < 1:
                parser.error("--depth must be at least 1")
            rows, added = extend_labels(conn, profiles, existing, encoder, args.model, args.variant, args.depth)
            save_labels(rows, args.labels)
            print(f"added {added} new rows; {len(rows)} total in {args.labels}")
            return 0
        if not existing:
            print(f"error: no labels in {args.labels}; run the pool command first", file=sys.stderr)
            return 1
        report = run_eval(conn, profiles, existing, encoder, args.model, args.variant, args.methods,
                          args.min_grade, args.only_reviewed)
    text = format_report(report)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

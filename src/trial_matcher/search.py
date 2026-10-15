"""Hybrid trial search: hard filters, then vector + full-text ranking merged with RRF.

    python -m trial_matcher.search "metastatic breast cancer, HER2 positive" --age 58 --sex FEMALE

Steps (all in Postgres, on trials that pass the filters):
  1. hard filters: status, sex (trial is ALL/unknown or matches), age (a missing limit is open);
  2. vector search: nearest trial embeddings to the query embedding;
  3. full-text search: trials sharing any word with the query (OR, stemmed);
  4. Reciprocal Rank Fusion: score = sum over lists of 1 / (60 + rank in that list).

Age and sex are explicit inputs; pulling them out of free text is a later step.
The query text is never logged.
"""

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import psycopg

from trial_matcher import config
from trial_matcher.db import connect
from trial_matcher.embed import BASIC, DEFAULT_MODEL, VARIANTS, Encoder, SentenceTransformerEncoder, vector_literal

DISCLAIMER = "Decision support, not medical advice."

OPEN_STATUSES = ("RECRUITING", "NOT_YET_RECRUITING")
SEXES = ("FEMALE", "MALE")  # the patient's sex; trials listing ALL accept either
MODES = ("hybrid", "vector", "fts")
DEFAULT_K = 50
POOL_SIZE = 200  # how deep each ranking goes before merging
RRF_K = 60  # standard RRF constant: damps the gap between rank 1 and rank 2

# bge models are trained to see this before a short search query (documents get no prefix).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class SearchError(Exception):
    """The search cannot run as asked (e.g. no embeddings for that model)."""


@dataclass(frozen=True)
class SearchResult:
    nct_id: str
    title: str
    status: str
    conditions: list[str]
    score: float  # RRF score (higher is better)
    vector_rank: int | None  # 1-based rank in the vector list, None if absent from it
    fts_rank: int | None  # same for the full-text list
    vector_distance: float | None  # cosine distance, smaller is closer


# --- pure helpers --------------------------------------------------------------


def query_prefix(model: str) -> str:
    return BGE_QUERY_PREFIX if model.startswith("BAAI/bge-") else ""


def fts_query_text(query: str, max_terms: int = 50) -> str:
    """Plain English -> a websearch_to_tsquery string meaning 'any of these words'.

    Only letters/digits survive, so quotes, minus signs and other operators in the input are inert.
    """
    seen: dict[str, None] = {}
    for token in re.findall(r"[a-z0-9]+", query.lower()):
        if len(token) >= 2 and token != "or":  # 'or' is the operator we join with
            seen.setdefault(token)
    return " or ".join(list(seen)[:max_terms])


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Merge ranked id lists (best first). Ties break on id so output is deterministic."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


# --- database queries ----------------------------------------------------------

# Hard filters, on table alias t. A NULL sex/age limit means "not restricted".
_FILTERS = """
    t.status = ANY(%(statuses)s)
    AND (t.sex IS NULL OR t.sex = 'ALL' OR t.sex = %(sex)s)
    AND (t.min_age_years IS NULL OR t.min_age_years <= %(age)s)
    AND (t.max_age_years IS NULL OR t.max_age_years >= %(age)s)
"""

_VECTOR_RANKING = f"""
SELECT e.nct_id, e.embedding <=> %(qvec)s::vector AS distance
FROM trial_embeddings e JOIN trials t USING (nct_id)
WHERE e.model = %(model)s AND e.variant = %(variant)s AND {_FILTERS}
ORDER BY distance, e.nct_id
LIMIT %(pool)s
"""

_FTS_RANKING = f"""
SELECT t.nct_id
FROM trials t, websearch_to_tsquery('english', %(fts)s) AS q
WHERE t.search_tsv @@ q AND {_FILTERS}
ORDER BY ts_rank(t.search_tsv, q) DESC, t.nct_id
LIMIT %(pool)s
"""


def search(
    conn: psycopg.Connection,
    query: str,
    *,
    age: float,
    sex: str,
    encoder: Encoder | None = None,
    model: str = DEFAULT_MODEL,
    variant: str = BASIC,
    k: int = DEFAULT_K,
    mode: str = "hybrid",
    statuses: Sequence[str] = OPEN_STATUSES,
    pool: int = POOL_SIZE,
) -> list[SearchResult]:
    """Top-k trials for a patient description. `encoder` is required unless mode='fts'."""
    sex = sex.upper()
    if sex not in SEXES:
        raise ValueError(f"sex must be one of {SEXES}, got {sex!r}")
    if not 0 <= age <= 130:
        raise ValueError(f"age must be between 0 and 130, got {age}")
    if k < 1 or pool < 1:
        raise ValueError("k and pool must be at least 1")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")

    params = {"statuses": list(statuses), "sex": sex, "age": age, "pool": pool}
    vector_rows: list[tuple[str, float]] = []
    fts_ids: list[str] = []

    if mode in ("hybrid", "vector"):
        if encoder is None:
            raise ValueError("an encoder is required for vector and hybrid search")
        has_vectors = conn.execute(
            "SELECT EXISTS (SELECT 1 FROM trial_embeddings WHERE model = %s AND variant = %s)", (model, variant)
        ).fetchone()[0]
        if not has_vectors:
            raise SearchError(
                f"no embeddings for model {model!r}, variant {variant!r}; "
                f"run: python -m trial_matcher.embed_trials --model {model} --variant {variant}"
            )
        [qvec] = encoder.encode([query_prefix(model) + query])
        try:
            vector_rows = conn.execute(
                _VECTOR_RANKING, {**params, "qvec": vector_literal(qvec), "model": model, "variant": variant}
            ).fetchall()
        except psycopg.errors.DataException as exc:  # e.g. "different vector dimensions"
            raise SearchError(f"query vector does not match the stored embeddings of {model!r}: {exc}") from exc

    fts_text = fts_query_text(query)
    if mode in ("hybrid", "fts") and fts_text:
        fts_ids = [r[0] for r in conn.execute(_FTS_RANKING, {**params, "fts": fts_text}).fetchall()]

    vector_ids = [nct for nct, _ in vector_rows]
    distance = dict(vector_rows)
    vector_rank = {nct: i for i, nct in enumerate(vector_ids, start=1)}
    fts_rank = {nct: i for i, nct in enumerate(fts_ids, start=1)}

    fused = reciprocal_rank_fusion([vector_ids, fts_ids])[:k]
    if not fused:
        return []
    details = {
        nct: (title, status, conditions)
        for nct, title, status, conditions in conn.execute(
            "SELECT nct_id, title, status, conditions FROM trials WHERE nct_id = ANY(%s)", ([n for n, _ in fused],)
        )
    }
    return [
        SearchResult(
            nct_id=nct,
            title=details[nct][0],
            status=details[nct][1],
            conditions=details[nct][2],
            score=score,
            vector_rank=vector_rank.get(nct),
            fts_rank=fts_rank.get(nct),
            vector_distance=distance.get(nct),
        )
        for nct, score in fused
    ]


# --- command line --------------------------------------------------------------


def format_results(results: Sequence[SearchResult], show: int) -> str:
    lines = []
    for i, r in enumerate(results[:show], start=1):
        ranks = f"vec #{r.vector_rank or '-'} / text #{r.fts_rank or '-'}"
        lines.append(f"{i:2d}. {r.nct_id}  score {r.score:.4f}  ({ranks})")
        lines.append(f"    {r.title[:110]}")
        if r.conditions:
            lines.append(f"    Conditions: {'; '.join(r.conditions)[:110]}")
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find recruiting trials for a plain-English patient description.")
    parser.add_argument("query", help='e.g. "metastatic breast cancer, HER2 positive, progressed on tamoxifen"')
    parser.add_argument("--age", type=float, required=True, help="patient age in years")
    parser.add_argument("--sex", required=True, type=str.upper, choices=SEXES)
    parser.add_argument("--show", type=int, default=10, help="how many titles to print (default %(default)s)")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="candidates to retrieve (default %(default)s)")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="embedding model (default: %(default)s)")
    parser.add_argument("--variant", choices=VARIANTS, default=BASIC)
    parser.add_argument("--database-url", default=None, help="default: $DATABASE_URL")
    args = parser.parse_args(argv)
    if not 0 <= args.age <= 130:
        parser.error("--age must be between 0 and 130")
    if args.show < 1 or args.k < 1:
        parser.error("--show and --k must be at least 1")

    encoder = SentenceTransformerEncoder(args.model) if args.mode != "fts" else None
    try:
        with connect(args.database_url or config.database_url()) as conn:
            results = search(
                conn, args.query, age=args.age, sex=args.sex, encoder=encoder,
                model=args.model, variant=args.variant, k=args.k, mode=args.mode,
            )
    except SearchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not results:
        print("No matching trials.\n\n" + DISCLAIMER)
        return 0
    print(format_results(results, args.show))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Embed every trial's text with a local model and store the vectors in Postgres.

    python -m trial_matcher.embed_trials [--model NAME] [--variant basic|with_inclusion] [--limit N]

Re-runnable: a trial is only embedded when it has no vector yet or its text changed.
Documents get no query instruction prefix (bge models only use it on search queries).
"""

import argparse
import logging
import sys
from dataclasses import dataclass

import psycopg

from trial_matcher import config
from trial_matcher.db import connect
from trial_matcher.embed import (
    BASIC,
    DEFAULT_MODEL,
    VARIANTS,
    Encoder,
    SentenceTransformerEncoder,
    build_trial_text,
    text_hash,
    vector_literal,
)

log = logging.getLogger("trial_matcher.embed_trials")

DEFAULT_CHUNK = 256  # texts embedded and committed together; progress is logged per chunk

_SELECT_TRIALS = """
SELECT t.nct_id, t.title, t.conditions, t.interventions,
       COALESCE((SELECT array_agg(x.text ORDER BY x.position) FROM (
           SELECT c.text, c.position FROM criteria c
           WHERE c.nct_id = t.nct_id AND c.kind = 'inclusion'
           ORDER BY c.position LIMIT 5) x), '{}') AS inclusion
FROM trials t
ORDER BY t.nct_id
"""

_UPSERT = """
INSERT INTO trial_embeddings (nct_id, model, variant, text_hash, embedding)
VALUES (%s, %s, %s, %s, %s::vector)
ON CONFLICT (nct_id, model, variant) DO UPDATE SET
    text_hash = EXCLUDED.text_hash, embedding = EXCLUDED.embedding, created_at = now()
"""


@dataclass
class EmbedStats:
    trials: int = 0  # trials considered (after --limit)
    embedded: int = 0  # new or changed, written this run
    unchanged: int = 0  # already up to date, skipped


def run_embedding(
    conn: psycopg.Connection,
    encoder: Encoder,
    model: str,
    variant: str = BASIC,
    limit: int | None = None,
    chunk_size: int = DEFAULT_CHUNK,
) -> EmbedStats:
    existing = dict(
        conn.execute(
            "SELECT nct_id, text_hash FROM trial_embeddings WHERE model = %s AND variant = %s",
            (model, variant),
        ).fetchall()
    )
    rows = conn.execute(_SELECT_TRIALS).fetchall()
    if limit is not None:
        rows = rows[:limit]

    stats = EmbedStats(trials=len(rows))
    todo: list[tuple[str, str, str]] = []  # (nct_id, text, hash)
    for nct_id, title, conditions, interventions, inclusion in rows:
        text = build_trial_text(variant, title, conditions, interventions, inclusion)
        digest = text_hash(text)
        if existing.get(nct_id) == digest:
            stats.unchanged += 1
        else:
            todo.append((nct_id, text, digest))
    log.info(
        "%s / %s: %d trials, %d to embed, %d already up to date",
        model, variant, stats.trials, len(todo), stats.unchanged,
    )

    for start in range(0, len(todo), chunk_size):
        chunk = todo[start : start + chunk_size]
        vectors = encoder.encode([text for _, text, _ in chunk])
        if len(vectors) != len(chunk):
            raise RuntimeError(f"encoder returned {len(vectors)} vectors for {len(chunk)} texts")
        with conn.transaction(), conn.cursor() as cur:
            cur.executemany(
                _UPSERT,
                [(nct, model, variant, digest, vector_literal(v)) for (nct, _, digest), v in zip(chunk, vectors)],
            )
        stats.embedded += len(chunk)
        log.info("embedded %d/%d", stats.embedded, len(todo))
    return stats


def summarize(conn: psycopg.Connection, model: str, variant: str) -> dict:
    """Counts to eyeball after a run: embeddings vs trials, vector sizes, missing trials."""
    row = conn.execute(
        """SELECT (SELECT count(*) FROM trials),
                  count(*), count(DISTINCT dim), min(dim)
           FROM trial_embeddings WHERE model = %s AND variant = %s""",
        (model, variant),
    ).fetchone()
    trials, embedded, dims, dim = row
    return {"trials": trials, "embedded": embedded, "missing": trials - embedded, "distinct_dims": dims, "dim": dim}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model name (default: %(default)s)")
    parser.add_argument("--variant", choices=VARIANTS, default=BASIC, help="which trial text to embed")
    parser.add_argument("--limit", type=int, default=None, help="only the first N trials by NCT id (for testing)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--database-url", default=None, help="default: $DATABASE_URL")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with connect(args.database_url or config.database_url()) as conn:
        # Fail fast on an empty database, before spending time loading the model.
        if conn.execute("SELECT count(*) FROM trials").fetchone()[0] == 0:
            log.error("no trials in the database; run trial_matcher.ingest first")
            return 1
        log.info("loading model %s ...", args.model)
        encoder = SentenceTransformerEncoder(args.model)
        stats = run_embedding(conn, encoder, args.model, args.variant, args.limit, args.chunk_size)
        summary = summarize(conn, args.model, args.variant)
    log.info("done: %d embedded, %d unchanged", stats.embedded, stats.unchanged)
    log.info("in database for this model+variant: %s", summary)
    return 0 if summary["distinct_dims"] <= 1 else 1


if __name__ == "__main__":
    sys.exit(main())

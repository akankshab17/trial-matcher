"""Postgres access for ingestion: run log, trial upsert, criteria replacement."""

import psycopg
from psycopg.types.json import Jsonb

from trial_matcher.parse_trial import ParsedTrial
from trial_matcher.split_criteria import Criterion


class RecordError(Exception):
    """Postgres rejected this one record (bad data); the rest of the run can continue."""


def connect(url: str) -> psycopg.Connection:
    # autocommit: each statement commits on its own, and conn.transaction() makes real transactions.
    return psycopg.connect(url, autocommit=True)


_UPSERT_TRIAL = """
INSERT INTO trials (
    nct_id, title, status, phase, study_type, conditions, interventions, sex,
    min_age_years, max_age_years, healthy_volunteers, eligibility_text, raw_json,
    last_update_posted, fetched_at
) VALUES (
    %(nct_id)s, %(title)s, %(status)s, %(phase)s, %(study_type)s, %(conditions)s,
    %(interventions)s, %(sex)s, %(min_age_years)s, %(max_age_years)s,
    %(healthy_volunteers)s, %(eligibility_text)s, %(raw_json)s,
    %(last_update_posted)s, now()
)
ON CONFLICT (nct_id) DO UPDATE SET
    title = EXCLUDED.title,
    status = EXCLUDED.status,
    phase = EXCLUDED.phase,
    study_type = EXCLUDED.study_type,
    conditions = EXCLUDED.conditions,
    interventions = EXCLUDED.interventions,
    sex = EXCLUDED.sex,
    min_age_years = EXCLUDED.min_age_years,
    max_age_years = EXCLUDED.max_age_years,
    healthy_volunteers = EXCLUDED.healthy_volunteers,
    eligibility_text = EXCLUDED.eligibility_text,
    raw_json = EXCLUDED.raw_json,
    last_update_posted = EXCLUDED.last_update_posted,
    fetched_at = now()
"""


class PostgresStore:
    def __init__(self, conn: psycopg.Connection):
        self._conn = conn

    def start_run(self, query: str) -> int:
        row = self._conn.execute(
            "INSERT INTO ingest_runs (query) VALUES (%s) RETURNING id", (query,)
        ).fetchone()
        return row[0]

    def update_run(
        self,
        run_id: int,
        *,
        trials_seen: int,
        trials_stored: int,
        trials_skipped: int,
        skipped: list[dict],
        status: str = "running",
        error: str | None = None,
    ) -> None:
        self._conn.execute(
            """UPDATE ingest_runs SET
                 trials_seen = %s, trials_stored = %s, trials_skipped = %s, skipped = %s,
                 status = %s, error = %s,
                 finished_at = CASE WHEN %s = 'running' THEN NULL ELSE now() END
               WHERE id = %s""",
            (trials_seen, trials_stored, trials_skipped, Jsonb(skipped), status, error, status, run_id),
        )

    def save_trial(self, trial: ParsedTrial, criteria: list[Criterion]) -> None:
        """Upsert the trial and replace its criteria, atomically. Raises RecordError on bad data."""
        params = {
            "nct_id": trial.nct_id,
            "title": trial.title,
            "status": trial.status,
            "phase": trial.phase,
            "study_type": trial.study_type,
            "conditions": trial.conditions,
            "interventions": trial.interventions,
            "sex": trial.sex,
            "min_age_years": trial.min_age_years,
            "max_age_years": trial.max_age_years,
            "healthy_volunteers": trial.healthy_volunteers,
            "eligibility_text": trial.eligibility_text,
            "raw_json": Jsonb(trial.raw_json),
            "last_update_posted": trial.last_update_posted,
        }
        try:
            with self._conn.transaction():
                self._conn.execute(_UPSERT_TRIAL, params)
                self._conn.execute("DELETE FROM criteria WHERE nct_id = %s", (trial.nct_id,))
                with self._conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO criteria (nct_id, kind, position, text) VALUES (%s, %s, %s, %s)",
                        [(trial.nct_id, c.kind, c.position, c.text) for c in criteria],
                    )
        except (psycopg.errors.DataError, psycopg.errors.IntegrityError) as exc:
            raise RecordError(f"{type(exc).__name__}: {exc}".strip()) from exc

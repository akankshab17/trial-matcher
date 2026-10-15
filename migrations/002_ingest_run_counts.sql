-- 002: record what each ingest run stored and skipped, and why it failed.

ALTER TABLE ingest_runs
    ADD COLUMN trials_stored   INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN trials_skipped  INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN skipped         JSONB   NOT NULL DEFAULT '[]',  -- [{"nct_id": ..., "reason": ...}]
    ADD COLUMN error           TEXT;

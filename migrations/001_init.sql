-- 001: initial schema. Public ClinicalTrials.gov data only.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE trials (
    nct_id              TEXT PRIMARY KEY,
    title               TEXT NOT NULL,
    status              TEXT NOT NULL,
    phase               TEXT[] NOT NULL DEFAULT '{}',
    study_type          TEXT NOT NULL,
    conditions          TEXT[] NOT NULL DEFAULT '{}',
    interventions       TEXT[] NOT NULL DEFAULT '{}',
    sex                 TEXT,                 -- ALL / FEMALE / MALE
    min_age_years       DOUBLE PRECISION,     -- NULL = no minimum ("N/A")
    max_age_years       DOUBLE PRECISION,     -- NULL = no maximum ("N/A")
    healthy_volunteers  BOOLEAN,
    eligibility_text    TEXT,
    raw_json            JSONB NOT NULL,
    last_update_posted  DATE,
    fetched_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE criteria (
    id        BIGSERIAL PRIMARY KEY,
    nct_id    TEXT NOT NULL REFERENCES trials (nct_id) ON DELETE CASCADE,
    kind      TEXT NOT NULL CHECK (kind IN ('inclusion', 'exclusion', 'unknown')),
    position  INTEGER NOT NULL,
    text      TEXT NOT NULL,
    UNIQUE (nct_id, position)
);

CREATE INDEX criteria_nct_id_idx ON criteria (nct_id);

CREATE TABLE ingest_runs (
    id            BIGSERIAL PRIMARY KEY,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ,
    query         TEXT NOT NULL,
    trials_seen   INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'done', 'failed'))
);

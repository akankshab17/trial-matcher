-- 003: trial embeddings (one row per trial x model x text variant) and full-text search column.

CREATE TABLE trial_embeddings (
    nct_id      TEXT NOT NULL REFERENCES trials (nct_id) ON DELETE CASCADE,
    model       TEXT NOT NULL,   -- e.g. 'BAAI/bge-small-en-v1.5'
    variant     TEXT NOT NULL,   -- which text was embedded: 'basic' | 'with_inclusion'
    text_hash   TEXT NOT NULL,   -- sha256 of the embedded text; unchanged hash = skip re-embedding
    -- No fixed dimension on purpose: 384-d and 768-d models share the table. Always filter by
    -- model + variant before comparing vectors. No vector index: exact search is ~5 ms at this size.
    embedding   vector NOT NULL,
    dim         INTEGER GENERATED ALWAYS AS (vector_dims(embedding)) STORED,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (nct_id, model, variant)
);

-- Full-text search over title + conditions + interventions. array_to_string is only STABLE, and
-- generated columns need IMMUTABLE, so wrap it (safe: text[] -> text has no locale dependence).
CREATE FUNCTION immutable_array_to_string(text[]) RETURNS text
    LANGUAGE sql IMMUTABLE PARALLEL SAFE
    AS $$ SELECT array_to_string($1, ' ') $$;

ALTER TABLE trials ADD COLUMN search_tsv tsvector GENERATED ALWAYS AS (
    to_tsvector('english',
        title || ' ' || immutable_array_to_string(conditions) || ' ' || immutable_array_to_string(interventions))
) STORED;

CREATE INDEX trials_search_tsv_idx ON trials USING GIN (search_tsv);

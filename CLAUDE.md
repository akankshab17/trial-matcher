# Trial Matcher

Matches plain-English patient profiles to relevant trials on ClinicalTrials.gov. Retrieves candidate trials, then judges eligibility against the profile and explains each match.

## Stack

- Python 3.12, FastAPI
- Postgres + pgvector (embeddings and relational data in one DB)
- Docker / docker compose for local dev
- pytest for tests

## Non-negotiable rules

1. **Public data only.** Only ClinicalTrials.gov and other public sources. Never store, log, or commit real patient data. Test profiles must be synthetic. Do not log raw profile text in production code paths.
2. **Disclaimer on all output.** Every API response and any user-facing rendering that contains match results must carry the label: **"Decision support, not medical advice."** This is part of the response schema, not an optional field, and is covered by tests.
3. **Small commits.** One logical change per commit, imperative message, tests included with the change. Don't bundle refactors with behavior changes.
4. **Tests for every module.** Each module in `src/` has a matching test file in `tests/`. New code without tests is not done. Use fixtures/recorded responses instead of live network calls in unit tests.
5. **Measure retrieval and LLM changes.** Any change to retrieval (chunking, embeddings, indexing, ranking, filters) or to any LLM prompt/model/parameter must be measured on the eval set in `/evals` before and after. Put the before/after numbers in the commit or PR description. No eval numbers, no merge.

## Layout (intended)

```
src/trial_matcher/   application code (api, ingest, retrieval, judge, schemas)
tests/               pytest, mirrors src/
evals/               eval sets, runners, and recorded results
docker/              Dockerfiles, compose files
```

Adjust this section as the structure becomes real; keep it accurate.

## Commands

```
docker compose up -d db                    # start Postgres + pgvector
./scripts/migrate.sh                       # apply migrations/*.sql (idempotent)
.venv/bin/python -m pytest                 # tests (DB tests use a throwaway Postgres via pgserver)
set -a; source .env; set +a
.venv/bin/python -m trial_matcher.ingest --max-trials 100   # small ingest; omit flag for all trials
.venv/bin/python scripts/criteria_fallback_stats.py         # criteria-splitter fallback rates
```

## Conventions

- Type hints throughout; validate I/O with Pydantic models.
- Config via environment variables; no secrets in the repo.
- DB schema changes go through migrations, never ad hoc DDL.
- Keep eval sets versioned and never tune on the test split.

## Roadmap — do NOT build yet

Stay within the current phase. Don't add scaffolding, dependencies, or abstractions for later phases.

- **Phase 1 (current):** ingest ClinicalTrials.gov, pgvector retrieval, LLM eligibility judging, FastAPI service, eval set in `/evals`.
- **Phase 2:** LangGraph agent, Langfuse tracing, CI eval gate.
- **Phase 3:** LoRA fine-tuned judge model.
- **Phase 4:** EKS deployment with Helm.

#!/usr/bin/env bash
# Apply migrations/*.sql in filename order, once each, inside the db container.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a

psql_db() { docker compose exec -T db psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" "$@"; }

psql_db -q -c "CREATE TABLE IF NOT EXISTS schema_migrations (
  filename TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now());"

for f in migrations/*.sql; do
  name=$(basename "$f")
  done_already=$(psql_db -tA -c "SELECT 1 FROM schema_migrations WHERE filename='$name'")
  if [ -n "$done_already" ]; then
    echo "skip   $name"
    continue
  fi
  echo "apply  $name"
  # one transaction per file: all of it or none of it
  { echo "BEGIN;"; cat "$f"; echo "INSERT INTO schema_migrations (filename) VALUES ('$name');"; echo "COMMIT;"; } | psql_db -q
done
echo "migrations up to date"

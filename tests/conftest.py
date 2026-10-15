import os
import uuid
from pathlib import Path

import psycopg
import pytest

MIGRATIONS = sorted((Path(__file__).parent.parent / "migrations").glob("*.sql"))


@pytest.fixture(scope="session")
def pg_url(tmp_path_factory):
    """A Postgres to test against.

    TEST_DATABASE_URL if set (e.g. the docker db); otherwise a throwaway local server via pgserver.
    Tests never touch real tables: each one gets its own schema (see `conn`).
    """
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        yield url
        return
    pgserver = pytest.importorskip("pgserver")
    server = pgserver.get_server(tmp_path_factory.mktemp("pgdata"), cleanup_mode="stop")
    yield server.get_uri()
    server.cleanup()


@pytest.fixture
def conn(pg_url):
    """Autocommit connection whose search_path is a fresh schema with all migrations applied."""
    schema = f"test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(pg_url, autocommit=True) as admin:
        admin.execute("CREATE EXTENSION IF NOT EXISTS vector")
        admin.execute(f"CREATE SCHEMA {schema}")
    connection = psycopg.connect(pg_url, autocommit=True, options=f"-c search_path={schema},public")
    for path in MIGRATIONS:
        connection.execute(path.read_text())
    yield connection
    connection.close()
    with psycopg.connect(pg_url, autocommit=True) as admin:
        admin.execute(f"DROP SCHEMA {schema} CASCADE")

"""Settings from environment variables (see .env.example)."""

import os


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env, then run: set -a; source .env; set +a"
        )
    return url

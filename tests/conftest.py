import base64
import json
import os
import sqlite3
import time
from contextlib import closing

import httpx
import pytest

# Provider tests should not fetch pricing metadata while importing LiteLLM.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    import dspy

    monkeypatch.setattr(
        dspy,
        "cache",
        dspy.clients.cache.Cache(
            disk_cache_dir=str(tmp_path / "cache"),
            enable_disk_cache=False,
            enable_memory_cache=True,
        ),
    )


@pytest.fixture
def codex_login(tmp_path):
    """Write `<tmp>/codex/auth.json` with an unsigned token, shaped like `codex login` output."""

    def login(**claims):
        claims = {
            "exp": time.time() + 3600,
            "https://api.openai.com/auth": {"chatgpt_account_id": "test-account"},
            **claims,
        }
        payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
        path = tmp_path / "codex" / "auth.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"tokens": {"access_token": f"header.{payload}.signature"}}))
        return path

    return login


@pytest.fixture
def codex_transport(monkeypatch):
    """Send Codex requests to a handler instead of the network."""

    def route(handler):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr("bird_gepa.codex._client", client)

    return route


@pytest.fixture
def demo(tmp_path):
    """Create eight synthetic questions; these are not BIRD benchmark results."""
    database = tmp_path / "demo" / "demo.sqlite"
    database.parent.mkdir()
    with closing(sqlite3.connect(database)) as conn:
        conn.execute("CREATE TABLE numbers (value INTEGER)")
        conn.executemany("INSERT INTO numbers VALUES (?)", [(n,) for n in range(1, 11)])
        conn.commit()
    return [
        {
            "example_id": f"demo:{n}",
            "db_id": "demo",
            "question": f"How many numbers are greater than {n}?",
            "evidence": "",
            "SQL": f"SELECT COUNT(*) FROM numbers WHERE value > {n}",
        }
        for n in range(1, 9)
    ]

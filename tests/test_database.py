import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Barrier

import pytest

from bird_gepa.data import locate_databases
from bird_gepa.database import Database, query_tool, trace_queries


@pytest.mark.parametrize("sql", ["DELETE FROM numbers", "ATTACH ':memory:' AS other", ""])
def test_query_rejects_non_select(tmp_path, demo, sql):
    database = Database(tmp_path / "demo/demo.sqlite")
    assert database.query(sql)[1] is not None
    assert database.query("SELECT COUNT(*) FROM numbers") == ([(10,)], None)


def test_database_primitive_bounds_results_and_binds_parameters(tmp_path, demo):
    rows = demo
    with closing(sqlite3.connect(tmp_path / "demo/demo.sqlite")) as conn:
        conn.executemany("INSERT INTO numbers VALUES (?)", [(n,) for n in range(11, 41)])
        conn.commit()
    databases = locate_databases(rows, tmp_path)
    query = query_tool(databases)
    assert query("missing", "SELECT 1")["error"]
    assert query("demo", "DELETE FROM numbers")["error"]
    assert "no such column" in query("demo", "SELECT missing FROM numbers")["error"]
    for limit in [-1, 0, 201, True]:
        assert query("demo", "SELECT 1", row_limit=limit)["error"]
    assert query("demo", "SELECT value FROM numbers ORDER BY value") == {
        "rows": [[n] for n in range(1, 21)],
        "truncated": True,
        "error": None,
    }
    assert query("demo", "SELECT value FROM numbers WHERE 0") == {
        "rows": [],
        "truncated": False,
        "error": None,
    }
    assert (
        query("demo", "SELECT sql FROM sqlite_master WHERE name = ?", ["numbers' OR 1=1 --"])[
            "rows"
        ]
        == []
    )
    assert query("demo", "SELECT COUNT(*) FROM numbers WHERE value > ?", [38])["rows"] == [[2]]
    assert query("demo", "SELECT x'00ff'")["rows"] == [[{"blob_hex": "00ff"}]]


@pytest.mark.parametrize("sql", ["SELECT SUM(value) FROM numbers", "SELECT missing FROM numbers"])
def test_each_query_closes_its_connection(tmp_path, demo, monkeypatch, sql):
    connections = []
    original = sqlite3.connect

    def connect(*args, **kwargs):
        connection = original(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    database = Database(tmp_path / "demo/demo.sqlite")
    assert connections == []
    for _ in range(2):
        database.query(sql)
    assert len(connections) == 2
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


def test_parallel_queries_have_independent_deadlines(tmp_path, demo, monkeypatch):
    barrier = Barrier(2)
    original = sqlite3.connect

    class Connection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql.startswith("WITH RECURSIVE"):
                # Both deadlines are installed before either query starts.
                barrier.wait(timeout=5)
            return super().execute(sql, parameters)

    monkeypatch.setattr(
        sqlite3, "connect", lambda *args, **kwargs: original(*args, factory=Connection, **kwargs)
    )
    database = Database(tmp_path / "demo/demo.sqlite")
    sql = (
        "WITH RECURSIVE numbers(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM numbers WHERE n<10000) "
        "SELECT SUM(n) FROM numbers"
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        expired = executor.submit(database.query, sql, timeout=0)
        valid = executor.submit(database.query, sql, timeout=5)
        assert expired.result() == ([], "interrupted")
        assert valid.result() == ([(50005000,)], None)


def test_query_can_follow_an_interrupted_query(tmp_path, demo, monkeypatch):
    sql = (
        "WITH RECURSIVE numbers(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM numbers WHERE n<10000) "
        "SELECT SUM(n) FROM numbers"
    )
    database = Database(tmp_path / "demo/demo.sqlite")
    clock = iter([0, 6])
    with monkeypatch.context() as patch:
        patch.setattr("bird_gepa.database.time.monotonic", lambda: next(clock, 6))
        assert database.query(sql)[1] == "interrupted"
    assert database.query(sql) == ([(50005000,)], None)


def test_tool_trace_counts_schema_queries_errors_and_rejected_attempts(tmp_path, demo):
    database = Database(tmp_path / "demo/demo.sqlite")
    query = query_tool({"demo": database})
    with trace_queries() as trace:
        query("demo", "SELECT name FROM sqlite_master")
        query("demo", "SELECT missing FROM numbers")
        query("missing", "SELECT 1")
        query("demo", "SELECT 1", row_limit=0)
        query("demo", "SELECT value FROM numbers WHERE value > ?", [2], row_limit=3)
        database.query("SELECT COUNT(*) FROM numbers")
    records = trace.records
    assert len(records) == 5
    assert [record.error is not None for record in records] == [False, True, True, True, False]
    assert records[-1].parameters == [2]
    assert records[-1].row_count == 3 and records[-1].truncated is True
    assert all(record.seconds >= 0 for record in records)
    query("demo", "SELECT 1")
    assert trace.count == 5

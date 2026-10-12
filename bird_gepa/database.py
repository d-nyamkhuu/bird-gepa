"""Read-only SQLite connections and the host tool available to Flex."""

import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}
MAX_ROW_LIMIT = 200
QUERY_TIMEOUT = 5.0


class QueryResult(TypedDict):
    """What `query_database` returns to the sandbox, as plain JSON-compatible values."""

    rows: list[list[Any]]
    truncated: bool
    error: str | None


@dataclass(frozen=True)
class QueryRecord:
    db_id: str
    sql: str
    parameters: list | None
    row_count: int
    truncated: bool
    error: str | None
    seconds: float


@dataclass
class QueryTrace:
    """The host tool attempts made during one solver invocation."""

    records: list[QueryRecord] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.records)

    @property
    def seconds(self) -> float:
        return sum(record.seconds for record in self.records)


_current_trace: ContextVar[QueryTrace | None] = ContextVar("current_trace", default=None)


@contextmanager
def trace_queries() -> Iterator[QueryTrace]:
    """Capture host tool attempts within one solver invocation."""
    trace = QueryTrace()
    token = _current_trace.set(trace)
    try:
        yield trace
    finally:
        _current_trace.reset(token)


def encode_blob(value: object) -> dict:
    """Represent SQLite BLOB values in JSON. Usable as `json.dump(default=...)`."""
    if isinstance(value, bytes):
        return {"blob_hex": value.hex()}
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def decode_blob(value: dict) -> object:
    """Reverse `encode_blob`. Usable as `json.load(object_hook=...)`."""
    return bytes.fromhex(value["blob_hex"]) if set(value) == {"blob_hex"} else value


def authorize(action: int, arg1: str | None, arg2: str | None, *rest: object) -> int:
    """Allow reads only. For SQLITE_FUNCTION, `arg2` is the function name."""
    if action in ALLOWED_ACTIONS and arg2 != "load_extension":
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


class Database:
    """A database file queried through independent read-only connections."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def query(
        self,
        sql: str,
        parameters: tuple = (),
        *,
        timeout: float = QUERY_TIMEOUT,
        limit: int | None = None,
    ) -> tuple[list[tuple], str | None]:
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("PRAGMA query_only = ON")
            connection.set_authorizer(authorize)
            deadline = time.monotonic() + timeout
            connection.set_progress_handler(lambda: time.monotonic() > deadline, 1000)
            try:
                with closing(connection.execute(sql, parameters)) as cursor:
                    if cursor.description is None:
                        return [], "Expected a SELECT query."
                    return (cursor.fetchall() if limit is None else cursor.fetchmany(limit)), None
            except sqlite3.Error as exc:
                return [], str(exc)


def query_tool(databases: dict[str, Database]) -> Callable[..., QueryResult]:
    """Build the `query_database` tool. Flex exposes it by this name and docstring."""

    def query_database(
        db_id: str, sql: str, parameters: list | None = None, row_limit: int = 20
    ) -> QueryResult:
        """Run read-only SQLite SQL with optional bound parameters and a five-second timeout.

        Return rows, a truncation flag, and an error. row_limit must be 1..200.
        BLOB cells are encoded as {"blob_hex": "..."}. No gold answers are available.
        """
        started = time.perf_counter()
        rows: list[tuple] = []
        truncated = False
        if db_id not in databases:
            error: str | None = "Unknown database."
        elif type(row_limit) is not int or not 1 <= row_limit <= MAX_ROW_LIMIT:
            error = f"row_limit must be 1..{MAX_ROW_LIMIT}."
        else:
            # Fetch one extra row to tell whether the result was truncated.
            rows, error = databases[db_id].query(sql, tuple(parameters or []), limit=row_limit + 1)
            rows, truncated = rows[:row_limit], len(rows) > row_limit

        # Rejected attempts are traced too, since they still cost the solver a query.
        trace = _current_trace.get()
        if trace is not None:
            trace.records.append(
                QueryRecord(
                    db_id=db_id,
                    sql=sql,
                    parameters=parameters,
                    row_count=len(rows),
                    truncated=truncated,
                    error=error,
                    seconds=time.perf_counter() - started,
                )
            )
        return {
            "rows": [
                [encode_blob(value) if isinstance(value, bytes) else value for value in row]
                for row in rows
            ],
            "truncated": truncated,
            "error": error,
        }

    return query_database

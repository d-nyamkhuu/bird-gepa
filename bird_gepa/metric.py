"""SQL correctness, query cost, and execution feedback for GEPA."""

import reprlib

import dspy

from bird_gepa.database import Database, QueryTrace

# Bump when scoring changes, so checkpoints from an older metric are not resumed.
METRIC_VERSION = 2

MAX_SQL_CHARS = 2000
MAX_TRACE_SQL_CHARS = 800
MAX_TRACE_RECORDS = 12
MAX_SAMPLE_ROWS = 3


def row_sample(rows: list[tuple], wanted: set[tuple]) -> str:
    """Show a few distinct rows, with bounded cell representations."""
    sample: list[tuple] = []
    for row in rows:
        if row in wanted and row not in sample:
            sample.append(row)
            if len(sample) == MAX_SAMPLE_ROWS:
                break
    return reprlib.repr(sample)


def row_differences(expected_rows: list[tuple], rows: list[tuple]) -> list[str]:
    """Describe how a wrong result differs from the expected one."""
    expected, actual = set(expected_rows), set(rows)
    missing, unexpected = expected - actual, actual - expected
    return [
        f"Missing distinct rows: {len(missing)}; sample: {row_sample(expected_rows, missing)}",
        f"Unexpected distinct rows: {len(unexpected)}; sample: {row_sample(rows, unexpected)}",
        f"Expected column counts: {sorted({len(row) for row in expected})}; "
        f"returned column counts: {sorted({len(row) for row in actual})}.",
    ]


def query_trace_lines(queries: QueryTrace) -> list[str]:
    """List the solver's host queries, bounded for the reflection prompt."""
    lines = ["Host query trace (grading SQL excluded):"]
    for index, record in enumerate(queries.records[:MAX_TRACE_RECORDS], start=1):
        lines.append(
            f"{index}. Database: {record.db_id}; SQL: {record.sql[:MAX_TRACE_SQL_CHARS]}; "
            f"parameters: {reprlib.repr(record.parameters)}; preview rows: {record.row_count}; "
            f"truncated: {record.truncated}; error: {reprlib.repr(record.error)}."
        )
    if queries.count > MAX_TRACE_RECORDS:
        omitted = queries.count - MAX_TRACE_RECORDS
        lines.append(f"... {omitted} more query attempts omitted from feedback.")
    return lines


class ExecutionMetric:
    """Execution accuracy with feedback for GEPA.

    Incorrect answers score zero. Correct answers pay `query_penalty` for each host
    query after the first.
    """

    def __init__(self, databases: dict[str, Database], *, query_penalty: float = 0.01) -> None:
        self.databases = databases
        self.query_penalty = query_penalty

    def __call__(
        self,
        gold: dspy.Example,
        pred: dspy.Prediction,
        trace: object = None,
        pred_name: object = None,
        pred_trace: object = None,
        program_trace: list[tuple] | None = None,
    ) -> dspy.Prediction:
        # GEPA's metric protocol. `program_trace` is passed only because it is declared.
        queries: QueryTrace = pred.query_trace
        sql = getattr(pred, "sql", "")
        rows, error = self.databases[gold.db_id].query(sql)
        expected, actual = set(gold.expected_rows), set(rows)
        correct = error is None and actual == expected
        extra_queries = max(0, queries.count - 1)
        score = max(0.0, 1.0 - self.query_penalty * extra_queries) if correct else 0.0

        if correct:
            outcome = "Correct."
        elif error:
            outcome = f"SQL execution failed: {error}"
        else:
            outcome = "Wrong result."
        if self.query_penalty:
            advice = "Reduce unnecessary queries while preserving correctness."
        else:
            advice = "Optimize execution accuracy. Query count does not affect the score."
        feedback = [
            outcome,
            f"Final SQL: {sql[:MAX_SQL_CHARS]}",
            f"Expected {len(gold.expected_rows)} rows ({len(expected)} distinct); "
            f"returned {len(rows)} rows ({len(actual)} distinct). "
            "Grading compares complete unordered row sets and ignores duplicate multiplicity.",
            f"Solver queries: {queries.count}; "
            f"additional queries after the first: {extra_queries}. "
            f"Correct-answer penalty: {self.query_penalty:g} per additional query. "
            f"Score: {score:.4f}. Incorrect answers always score zero. {advice}",
        ]
        if program_trace is not None:
            feedback.append(f"Predictor trace calls: {len(program_trace)} (not penalized).")
        if not correct:
            feedback += row_differences(gold.expected_rows, rows)
            feedback.append(f"Reference SQL: {gold.reference_sql[:MAX_SQL_CHARS]}")
        feedback += query_trace_lines(queries)
        return dspy.Prediction(
            score=score,
            correct=correct,
            query_count=queries.count,
            feedback="\n".join(feedback),
        )

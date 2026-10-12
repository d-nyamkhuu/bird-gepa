import inspect

import dspy
import pytest

from bird_gepa.database import Database, query_tool, trace_queries
from bird_gepa.metric import ExecutionMetric


def prediction(databases, sql, query_count):
    query = query_tool(databases)
    with trace_queries() as trace:
        for _ in range(query_count):
            query("demo", "SELECT 1")
    return dspy.Prediction(sql=sql, query_trace=trace)


@pytest.mark.parametrize(
    ("query_count", "penalty", "score"),
    [
        (0, 0.01, 1.0),
        (1, 0.01, 1.0),
        (4, 0.01, 0.97),
        (4, 0.02, 0.94),
        (4, 0.0, 1.0),
        (102, 0.01, 0.0),
    ],
)
def test_correct_answers_pay_for_queries_after_the_first(
    tmp_path, demo, query_count, penalty, score
):
    databases = {"demo": Database(tmp_path / "demo/demo.sqlite")}
    gold = dspy.Example(
        db_id="demo", expected_rows=[(10,)], reference_sql="SELECT COUNT(*) FROM numbers"
    )
    pred = prediction(databases, gold.reference_sql, query_count)
    result = ExecutionMetric(databases, query_penalty=penalty)(gold, pred)
    assert result.score == pytest.approx(score)
    assert result.correct is True
    assert result.query_count == query_count
    assert "Correct." in result.feedback
    assert "Reference SQL:" not in result.feedback
    if penalty == 0:
        assert "Optimize execution accuracy." in result.feedback
        assert "Reduce unnecessary queries" not in result.feedback
    else:
        assert "Reduce unnecessary queries" in result.feedback
    # The final grading query must not appear in the solver measurements.
    assert pred.query_trace.count == query_count


@pytest.mark.parametrize("sql", ["SELECT missing FROM numbers", "SELECT 99", "SELECT 10, 20"])
def test_failed_queries_and_wrong_results_have_diagnostic_feedback(tmp_path, demo, sql):
    databases = {"demo": Database(tmp_path / "demo/demo.sqlite")}
    gold = dspy.Example(
        db_id="demo", expected_rows=[(10,)], reference_sql="SELECT COUNT(*) FROM numbers"
    )
    pred = prediction(databases, sql, 4)
    result = ExecutionMetric(databases)(gold, pred, program_trace=[(), ()])
    assert result.score == 0
    assert result.correct is False
    assert "Final SQL: " + sql in result.feedback
    assert "Reference SQL: " + gold.reference_sql in result.feedback
    assert "Missing distinct rows: 1; sample: [(10,)]" in result.feedback
    assert "Solver queries: 4" in result.feedback
    assert "Predictor trace calls: 2 (not penalized)." in result.feedback
    assert "1. Database: demo; SQL: SELECT 1" in result.feedback
    if "missing" in sql:
        assert "SQL execution failed: no such column: missing" in result.feedback
    elif "," in sql:
        assert "Expected column counts: [1]; returned column counts: [2]" in result.feedback
    else:
        assert "Unexpected distinct rows: 1; sample: [(99,)]" in result.feedback


def test_empty_results_and_duplicate_rows_keep_existing_set_semantics(tmp_path, demo):
    databases = {"demo": Database(tmp_path / "demo/demo.sqlite")}
    for sql, expected in [("SELECT 1 WHERE 0", []), ("SELECT 1 UNION ALL SELECT 1", [(1,)])]:
        gold = dspy.Example(db_id="demo", expected_rows=expected, reference_sql=sql)
        result = ExecutionMetric(databases)(gold, prediction(databases, sql, 1))
        assert result.correct is True
        assert result.score == 1


def test_feedback_is_bounded_and_does_not_change_complete_result_grading(tmp_path, demo):
    databases = {"demo": Database(tmp_path / "demo/demo.sqlite")}
    gold = dspy.Example(
        db_id="demo",
        expected_rows=[("x" * 100000,)] * 1000,
        reference_sql="SELECT '" + "x" * 100000 + "'",
    )
    result = ExecutionMetric(databases)(gold, prediction(databases, "SELECT 1", 30))
    assert result.correct is False
    assert result.query_count == 30
    assert "... 18 more query attempts omitted" in result.feedback
    assert len(result.feedback) < 6000


def test_missing_host_measurements_cannot_receive_a_free_query_score(tmp_path, demo):
    databases = {"demo": Database(tmp_path / "demo/demo.sqlite")}
    gold = dspy.Example(db_id="demo", expected_rows=[(1,)], reference_sql="SELECT 1")
    with pytest.raises(AttributeError, match="query_trace"):
        ExecutionMetric(databases)(gold, dspy.Prediction(sql="SELECT 1", query_count=0))


def test_metric_follows_the_gepa_feedback_protocol(tmp_path, demo):
    metric = ExecutionMetric({"demo": Database(tmp_path / "demo/demo.sqlite")})
    # GEPA binds five positional arguments, and passes `program_trace` only when declared.
    inspect.signature(metric).bind(None, None, None, None, None)
    assert "program_trace" in inspect.signature(metric).parameters

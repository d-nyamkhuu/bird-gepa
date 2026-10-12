from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import dspy
import pytest

from bird_gepa.database import Database, QueryTrace, query_tool, trace_queries
from bird_gepa.program import TracedFlex


def test_parallel_forwards_keep_host_query_counts_separate(tmp_path, demo, monkeypatch):
    query = query_tool({"demo": Database(tmp_path / "demo/demo.sqlite")})
    barrier = Barrier(2)

    def forward(self, *, count):
        query("demo", "SELECT 1")
        barrier.wait(timeout=5)
        for _ in range(count - 1):
            query("demo", "SELECT missing FROM numbers")
        # Editable source must not be able to claim a lower query count.
        return dspy.Prediction(sql="SELECT 1", query_trace=QueryTrace())

    monkeypatch.setattr(dspy.Flex, "forward", forward)
    solver = TracedFlex("count -> sql", tools=[query], max_predictor_calls=10)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda count: solver(count=count), [2, 5]))
    assert [result.query_trace.count for result in results] == [2, 5]
    for result in results:
        records = result.query_trace.records
        assert records[0].error is None
        assert all("no such column" in record.error for record in records[1:])
        assert result.query_trace.seconds >= 0


def test_failed_forward_restores_the_enclosing_query_trace(tmp_path, demo, monkeypatch):
    query = query_tool({"demo": Database(tmp_path / "demo/demo.sqlite")})

    def forward(self, **inputs):
        query("demo", "SELECT 1")
        raise ValueError("Failed solver")

    monkeypatch.setattr(dspy.Flex, "forward", forward)
    solver = TracedFlex("question -> sql", tools=[query])
    with trace_queries() as outer:
        with pytest.raises(ValueError, match="Failed solver"):
            solver(question="test")
        assert outer.records == []
        query("demo", "SELECT 2")
        assert outer.count == 1
        assert outer.records[0].sql == "SELECT 2"

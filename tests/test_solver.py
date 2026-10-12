import ast
import inspect
import shutil
import sqlite3
from contextlib import closing

import dspy
import pytest
from dspy.utils import DummyLM

from bird_gepa import seed
from bird_gepa.data import build_examples, locate_databases
from bird_gepa.database import query_tool
from bird_gepa.program import SEED_SOURCE, SIGNATURE, TracedFlex


def test_seed_declares_the_host_tool_signature():
    stub = next(
        node
        for node in ast.walk(ast.parse(inspect.getsource(seed)))
        if isinstance(node, ast.FunctionDef) and node.name == "query_database"
    )
    parameters = inspect.signature(query_tool({})).parameters.values()
    assert [arg.arg for arg in stub.args.args] == [parameter.name for parameter in parameters]
    assert [ast.literal_eval(default) for default in stub.args.defaults] == [
        parameter.default for parameter in parameters if parameter.default is not parameter.empty
    ]


@pytest.mark.skipif(shutil.which("deno") is None, reason="Flex integration requires Deno")
def test_seed_helpers_are_editable_and_schema_discovery_paginates(tmp_path, demo):
    rows = demo
    with closing(sqlite3.connect(tmp_path / "demo/demo.sqlite")) as conn:
        for n in range(205):
            conn.execute(f"CREATE TABLE extra_{n} (secret TEXT)")
        conn.commit()
    # Keep the seed helper implementations, but return their observations for assertions.
    source = (
        SEED_SOURCE[: SEED_SOURCE.index("    def forward(")]
        + """
    def forward(self, db_id, **inputs):
        tables = self.list_tables(db_id)
        return dspy.Prediction(sql=self.inspect_schema(db_id, tables), tables=tables)
"""
    )
    databases = locate_databases(rows, tmp_path)
    examples = build_examples(rows, databases)
    solver = dspy.Flex(SIGNATURE, tools=[query_tool(databases)], max_predictor_calls=3)
    solver.load_state({"module_src": source, "lm": None})
    result = solver(**examples[0].inputs())
    assert len(result.tables) == 206
    assert result.sql.count("CREATE TABLE") == 206
    # Edit the discovery helper itself, as a GEPA source proposal can.
    source = source.replace("AND name NOT GLOB 'sqlite_*'", "AND name = 'numbers'")
    solver.load_state({"module_src": source, "lm": None})
    result = solver(**examples[0].inputs())
    assert result.tables == ["numbers"]
    assert "value INTEGER" in result.sql and "secret" not in result.sql


@pytest.mark.skipif(shutil.which("deno") is None, reason="Flex integration requires Deno")
@pytest.mark.parametrize("initial_sql", ["SELECT missing FROM numbers", "SELECT 0"])
def test_seed_repairs_sql_from_execution_feedback(tmp_path, demo, initial_sql):
    lm = DummyLM(
        [
            {"reasoning": "Initial attempt.", "sql": initial_sql},
            {
                "reasoning": "Use the available column and requested count.",
                "sql": "SELECT COUNT(*) FROM numbers WHERE value > 1",
            },
        ],
        adapter=dspy.JSONAdapter(),
    )
    databases = locate_databases(demo, tmp_path)
    examples = build_examples(demo, databases)
    solver = dspy.Flex(SIGNATURE, tools=[query_tool(databases)], max_predictor_calls=3)
    solver.load_state({"module_src": SEED_SOURCE, "lm": None})
    solver.set_lm(lm)
    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
        prediction = solver(**examples[0].inputs())
    assert databases["demo"].query(prediction.sql) == ([(9,)], None)
    assert len(lm.history) == 2
    repair_prompt = str(lm.history[-1]["messages"])
    assert initial_sql in repair_prompt
    assert ("no such column" if "missing" in initial_sql else "[[0]]") in repair_prompt


@pytest.mark.skipif(shutil.which("deno") is None, reason="Flex integration requires Deno")
@pytest.mark.parametrize(("calls", "succeeds"), [(10, True), (11, False)])
def test_predictor_limit_allows_ten_calls_and_rejects_the_eleventh(calls, succeeds):
    source = f"""
class SQLSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question -> sql")

    def forward(self, **inputs):
        for index in range({calls}):
            result = self.predict(question=str(index))
        return dspy.Prediction(sql=result.sql)
"""
    lm = DummyLM([{"sql": "SELECT 1"}] * calls, adapter=dspy.JSONAdapter())
    solver = TracedFlex("question -> sql", max_predictor_calls=10)
    solver.load_state({"module_src": source, "lm": None})
    solver.set_lm(lm)
    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
        if succeeds:
            result = solver(question="test")
            assert result.sql == "SELECT 1"
            assert result.query_trace.count == 0
        else:
            with pytest.raises(dspy.CodeInterpreterError, match="predictor-call budget.*10"):
                solver(question="test")
    assert len(lm.history) == 10

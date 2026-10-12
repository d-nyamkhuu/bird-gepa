"""The Flex program that GEPA optimizes, and the loader for a saved solver."""

import inspect
from pathlib import Path
from typing import Any

import dspy

from bird_gepa.data import INPUT_FIELDS
from bird_gepa.database import Database, query_tool, trace_queries
from bird_gepa.seed import SQLSolver

SIGNATURE = f"{', '.join(INPUT_FIELDS)} -> sql"
SEED_SOURCE = inspect.getsource(SQLSolver)
MAX_PREDICTOR_CALLS = 10


class TracedFlex(dspy.Flex):
    """Keep query measurements outside the optimizer-authored source."""

    def forward(self, **inputs: Any) -> dspy.Prediction:
        # Solvers are scored with the JSON adapter, so a reused solver runs with it too.
        with dspy.context(adapter=dspy.JSONAdapter()), trace_queries() as trace:
            prediction = super().forward(**inputs)
        prediction.query_trace = trace
        return prediction


def build_program(
    databases: dict[str, Database], task_lm: dspy.BaseLM, max_predictor_calls: int
) -> TracedFlex:
    """The seed solver, with the database tool and the task model attached."""
    program = TracedFlex(
        SIGNATURE, tools=[query_tool(databases)], max_predictor_calls=max_predictor_calls
    )
    # Flex takes its source through state; there is no constructor argument for it.
    program.load_state({"module_src": SEED_SOURCE, "lm": None})
    program.set_lm(task_lm)
    return program


def load_solver(
    path: Path | str,
    databases: Path | str,
    lm: dspy.BaseLM,
    *,
    max_predictor_calls: int = MAX_PREDICTOR_CALLS,
) -> TracedFlex:
    """Load a solver saved by the CLI, ready to answer questions.

    `databases` is a directory of `<db_id>/<db_id>.sqlite` files. Call the solver
    with `db_id`, `question` and `evidence`, and read the query from `.sql`.
    """
    directory = Path(databases)
    found = {
        file.parent.name: Database(file)
        for file in sorted(directory.glob("*/*.sqlite"))
        if file.stem == file.parent.name
    }
    if not found:
        raise ValueError(f"No <db_id>/<db_id>.sqlite databases in {directory}.")
    program = build_program(found, lm, max_predictor_calls)
    program.load(path)
    # The saved state holds no model, so loading clears the one attached above.
    program.set_lm(lm)
    return program

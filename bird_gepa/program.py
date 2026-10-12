"""The Flex program that GEPA optimizes: the seed solver source, run with host measurements."""

import inspect
from typing import Any

import dspy

from bird_gepa.data import INPUT_FIELDS
from bird_gepa.database import Database, query_tool, trace_queries
from bird_gepa.seed import SQLSolver

SIGNATURE = f"{', '.join(INPUT_FIELDS)} -> sql"
SEED_SOURCE = inspect.getsource(SQLSolver)


class TracedFlex(dspy.Flex):
    """Keep query measurements outside the optimizer-authored source."""

    def forward(self, **inputs: Any) -> dspy.Prediction:
        with trace_queries() as trace:
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

# BIRD SQL with DSPy Flex + GEPA

Optimize a text-to-SQL solver on [BIRD](https://bird-bench.github.io/) with
[DSPy Flex](https://dspy.ai/current/api/modules/Flex/) and
[GEPA](https://dspy.ai/current/api/optimizers/GEPA/overview/), using your
ChatGPT Codex login for model calls.

The solver is a plain Python class. Flex runs its source in a sandbox, and GEPA
rewrites that source (helpers, prompts and control flow) based on execution feedback.
Model weights stay fixed.

- [BIRD23 train-filtered](https://huggingface.co/datasets/birdsql/bird23-train-filtered):
  6,601 questions across 69 databases, split by database into train and validation.
- [BIRD mini-dev](https://github.com/bird-bench/mini_dev): 500 SQLite questions
  across 11 other databases, held out for evaluation.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- [Deno](https://deno.com/), for the Flex sandbox
- Bash, jq and unzip, for downloading data
- A ChatGPT Codex login (`codex login`). No OpenAI API key is needed.

## Quickstart

```bash
uv sync
bash scripts/download_bird.sh
uv run python -m bird_gepa.prepare
uv run python -m bird_gepa.cli --output artifacts/bird-solver/solver.py
```

The download is about 9.7 GB, and the extracted databases take about 34 GB.
The run logs execution accuracy on the full validation split before and after
optimization, and writes the selected solver source to `--output`. Rerun the
same command to resume from its checkpoint.

```bash
uv run python -m bird_gepa.cli --help
```

## Layout

```text
bird_gepa/
├── seed.py          # The editable SQLSolver class that GEPA rewrites
├── database.py      # Read-only SQLite access and the query_database host tool
├── program.py       # The Flex program: seed source, host tool and query tracing
├── metric.py        # Execution accuracy, query penalty and feedback
├── optimization.py  # GEPA search, checkpoints and before/after evaluation
├── data.py          # Prepared splits and DSPy examples
├── codex.py         # A DSPy LM that sends requests with your Codex login
├── cli.py           # python -m bird_gepa.cli
└── prepare.py       # python -m bird_gepa.prepare
scripts/
└── download_bird.sh # Download questions and databases
tests/
```

Start with [seed.py](bird_gepa/seed.py), then [program.py](bird_gepa/program.py) and
[optimization.py](bird_gepa/optimization.py).

## Documentation

- [Data preparation](docs/DATA_PREPARATION.md): sources, cleaning, splitting and outputs.
- [Architecture](docs/IMPLEMENTATION_SPEC.md): the solver, database boundary, metric,
  GEPA setup, Codex model, CLI options and checkpoints.
- [Optimisation walkthrough](OPTIMISATION.md): what happens during a run, with results
  from a completed run.

## Development

```bash
uv run pytest -q
uv run pytest -q -m "not slow"  # skip the end-to-end Flex and GEPA runs
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

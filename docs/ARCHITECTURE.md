# Architecture

This covers what the README leaves out: how the pieces are wired, and the
choices behind them.

## The sandbox

[program.py](../bird_gepa/program.py) reads the solver with
`inspect.getsource(SQLSolver)` and loads it into a `TracedFlex`, a `dspy.Flex`
subclass. Flex runs that source in DSPy's Deno/Pyodide sandbox and injects
`query_database`. The `TYPE_CHECKING` declaration in [seed.py](../bird_gepa/seed.py)
only exists for editors and type checkers.

The helpers `list_tables`, `inspect_schema` and `preview_sql` are part of the
editable source. Table discovery is paginated and schema is fetched in batches,
so the 200-row cap never truncates a large schema silently.

The ten-call cap counts calls Flex bridges to the host. Steps inside a `ReAct`
or RLM predictor are not counted.

## The database boundary

The limits on `query_database` live on the host, where the solver cannot edit
them. [database.py](../bird_gepa/database.py) enforces read-only SQL with an
SQLite authorizer, rejects unknown database IDs, and fetches one row past
`row_limit` to set `truncated`. Each query opens and closes its own read-only
connection, so concurrent queries get independent deadlines and need no pool.

`trace_queries` collects one solver call's queries in a `ContextVar`.
`TracedFlex.forward` attaches the resulting `QueryTrace` (`records`, `count`,
`seconds`) to the prediction, overriding anything the solver set. Every attempt
counts, including schema reads, errors and rejected arguments. Preparation and
grading call `Database.query` directly and are not counted.

## Metric feedback

`ExecutionMetric` follows GEPA's feedback-metric protocol. Because the score is
one number, a candidate can trade some accuracy for fewer queries across a
batch. With a positive penalty, an all-correct minibatch still leaves room to
improve, so it can still trigger reflection.

Feedback holds the final SQL, complete and distinct row counts, the score and
up to twelve query records with their SQL, parameters, errors and truncation
flags. Wrong answers add the reference SQL, both column counts, and up to three
missing and three unexpected rows. Only feedback is truncated; grading always
compares complete results.

`METRIC_VERSION` is stored in checkpoints. Bump it when scoring changes.

## The selection subset and budget

The subset is drawn once with the run seed. Slots are allocated to databases by
largest remainders, so with the default data all 15 validation databases are
covered. GEPA averages question scores without weighting by database, and runs
no periodic full validation.

The budget counts GEPA's metric calls only, not the before and after passes, and
is approximate at batch boundaries. With `--skip-baseline` GEPA still scores the
seed on the subset.

`optimize` returns an `OptimizationResult`: the program and an
`EvaluationSummary` per full pass (`baseline` is `None` when skipped). A
summary's `accuracy` is raw execution accuracy. GEPA's own logs show the
penalised score.

## The saved solver

DSPy saves a Flex program as its `module_src`. Tools and the interpreter are
runtime dependencies and are not serialized, so `load_solver` rebuilds the same
`TracedFlex` with a `query_database` tool over the given directory, then loads
the state. `TracedFlex.forward` sets the JSON adapter, so a loaded solver formats
prompts as it did when it was scored. The model is cleared before saving.

## The Codex model

[codex.py](../bird_gepa/codex.py) sends Responses API requests to
`chatgpt.com/backend-api/codex/responses` with the login from `codex login`.

- It reads `$CODEX_HOME/auth.json` on every request and never writes it. Refresh
  tokens are single-use, so refreshing here would sign out the Codex CLI.
  Run `codex login` again when the token expires.
- The account ID and data-residency region come from the token's claims.
- Requests use `store=False` and `stream=True`. Codex rejects sampling options
  and output limits, so DSPy's `temperature` and `max_tokens` are dropped.
- Rate limits, server errors and truncated streams are retried three times with
  backoff. Other 4xx responses fail at once. Only completed responses are cached.

The task model has a 90-second timeout and a disk cache. The reflection model
has 180 seconds and no cache.

## Checkpoints

GEPA checkpoints at iteration boundaries, so an interrupted iteration may
repeat. Budgets are cumulative: raise `--metric-calls` to continue a finished run.

`checkpoint/manifest.json` records everything that affects scores: the training
and subset question IDs and inputs, model IDs, seed source, `SearchSettings`
and metric version. The budget and thread count are left out. A resume that
does not match, or a checkpoint without a manifest, fails before any model call.

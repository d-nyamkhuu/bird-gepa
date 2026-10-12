# Architecture

The modules are small and take explicit arguments. `cli.py` sets up a run, but
the solver and optimizer can also be called directly from Python.
`optimization.optimize` takes separate training and validation examples, and the
settings that affect scores as a `SearchSettings`. The CLI validates option
values, and `optimize` checks that both splits are nonempty and that the metric
budget exceeds the validation subset.

## Reading order

1. [seed.py](../bird_gepa/seed.py): the SQL workflow GEPA edits.
2. [database.py](../bird_gepa/database.py): the host database boundary.
3. [program.py](../bird_gepa/program.py): the Flex program, with host measurements.
4. [metric.py](../bird_gepa/metric.py): correctness, query cost and feedback.
5. [optimization.py](../bird_gepa/optimization.py): evaluation, checkpoints and GEPA.
6. [data.py](../bird_gepa/data.py): prepared splits and examples.
7. [cli.py](../bird_gepa/cli.py): options, logging and models.
8. [codex.py](../bird_gepa/codex.py): the Codex language model.

## Editable solver

`seed.py` defines an ordinary `SQLSolver(dspy.Module)`. Its `forward()`:

1. Lists the tables and fetches their schema.
2. Drafts SQL with a `ChainOfThought` predictor.
3. Executes the draft and collects an error or a result preview.
4. Asks a second predictor to review the SQL against the question, evidence,
   schema and preview, keeping it when it is already correct, and returns it.

The metric executes the returned SQL, so the seed does not run it again.

The helpers `list_tables`, `inspect_schema` and `preview_sql` are part of the
editable source too. Table discovery is paginated and schema is fetched in
batches, so the 200-row tool cap never truncates a large schema silently.

`program.py` reads the class source with `inspect.getsource(SQLSolver)` and
loads it into a `TracedFlex` (a Flex subclass). Flex instantiates and runs
that source in DSPy's Deno/Pyodide sandbox, with `query_database` injected. The
`TYPE_CHECKING` declaration in `seed.py` only documents the tool for editors and
type checkers.

GEPA can replace the whole class: helper implementations, prompts, branching and
repair strategy. Each question allows up to ten bridged predictor calls
(`--max-predictor-calls`). The seed uses two. The cap does not bound calls made
inside an RLM or ReAct predictor.

## Database boundary

`database.py` provides the one host tool:

```python
query_database(db_id, sql, parameters=None, row_limit=20)
```

It accepts only known database IDs. It enforces read-only SQL through an SQLite
authorizer, a five-second deadline and a `row_limit` of 1 to 200. It returns
`rows`, `truncated` and `error`, and encodes BLOB cells as
`{"blob_hex": "..."}`. It fetches one extra row to detect truncation. The tool
never sees reference SQL or expected answers. These limits live on the host,
outside the editable source.

`data.locate_databases` returns paths without opening anything. Each
`Database.query` opens its own read-only connection and closes it afterwards, so
concurrent queries have independent deadlines and need no locks or pool.

`trace_queries` uses a `ContextVar` to collect the queries of one solver call.
`TracedFlex.forward` attaches it to each prediction as `query_trace`, overriding
anything set by the editable source. A `QueryTrace` holds the `records`, with
their `count` and total `seconds`.

Every tool attempt counts, including schema reads, errors and rejected
arguments. Preparation and grading call `Database.query` directly and are not
counted. Query time covers connection and tool overhead but not model latency.
It is reported but not scored.

## Metric

[metric.py](../bird_gepa/metric.py) defines `ExecutionMetric`, a callable that
follows GEPA's feedback-metric protocol. It runs the final SQL and compares its complete,
unordered row set with the reference answer. Duplicate rows are ignored. This is
simpler than the official BIRD evaluator.

Incorrect answers score zero. Correct answers score
`max(0, 1 - query_penalty * max(0, query_count - 1))`, so the first query is free.
With the default penalty of 0.01:

- one query scores 1.00
- three queries, as the seed makes, score 0.98
- four queries score 0.97
- eight queries score 0.93

Because the score is a single number, fewer queries can outweigh some lost
accuracy across a batch. `--query-penalty 0` restores binary scoring. With a
positive penalty, an all-correct minibatch can still trigger reflection.

Feedback for GEPA includes:

- the final SQL
- complete and distinct row counts
- the query count and score
- up to twelve host query records, with their SQL, parameters, errors and
  truncation flags
- the predictor trace count, which is reported but not penalised

Wrong answers also get the reference SQL, expected and actual column counts, and
up to three missing and three unexpected distinct rows. Only the feedback is
truncated. Grading always uses complete results.

`METRIC_VERSION` is recorded in checkpoints. Bump it when scoring changes.

## Optimization

[GEPA](https://github.com/gepa-ai/gepa) is an evolutionary search guided by an LLM
that reflects on execution feedback. A candidate is the solver's source. GEPA
0.1.4 uses Pareto candidate selection, with merging disabled. `optimization.optimize`:

1. Evaluates the seed on the full validation split, unless `--skip-baseline`.
2. Picks a candidate from the Pareto frontier. Candidates that are best on some
   validation questions stay available even when another has higher overall
   accuracy.
3. Runs it on a reflection minibatch of training questions (eight by default)
   and collects traces, scores and feedback.
4. Asks the reflection model to propose revised solver source.
5. Runs the revision on the same minibatch. If its total improves, scores it on
   the validation subset and updates the frontier.
6. Repeats until the metric-call budget runs out, then evaluates the best
   candidate on the full validation split.

The validation subset (300 questions by default, `--validation-size`) is drawn
once, with the run seed and without replacement. Slots are allocated in
proportion to each database's share of the validation split, using largest
remainders. With the default data, the subset covers all 15 validation
databases. Very small subsets can leave databases out. GEPA averages question
scores without weighting by database, and runs no periodic full validation.

The budget counts GEPA's metric calls only, not the before and after evaluation,
and it is approximate at batch boundaries. `--skip-baseline` saves 1,376
evaluations. GEPA still scores the seed on the subset.

`optimize` returns an `OptimizationResult`: the optimized program, and an
`EvaluationSummary` for each full-validation pass (`baseline` is `None` when
skipped). A summary holds the correct count, the penalised score, query counts
and query time; its `accuracy` is raw execution accuracy, in percent. The CLI
writes the program's source to `--output`.
GEPA's own logs show the penalised score.

## Codex model

[codex.py](../bird_gepa/codex.py) defines `CodexLM`, a DSPy `BaseLM` that sends
Responses API requests to `https://chatgpt.com/backend-api/codex/responses` with
the ChatGPT login saved by `codex login`. It follows the same approach as Hermes
Agent:

- **Login file.** On every request it reads `$CODEX_HOME/auth.json` (default
  `~/.codex/auth.json`). It never writes the file, and never stores the token
  on the LM.
- **No refresh.** Refresh tokens are single-use, so refreshing here would sign
  out the Codex CLI. Run `codex login` when the login expires. An expired token
  fails before any request is sent.
- **Headers.** The account ID and any data-residency region come from the access
  token's claims. Requests identify themselves as `bird-gepa` rather than as the
  Codex CLI.
- **Request shape.** System messages become `instructions`, and requests use
  `store=False` and `stream=True`. Codex rejects output-token limits and sampling
  options, so DSPy's `temperature` and `max_tokens` are accepted but not sent.
  `reasoning_effort` is passed through, and any other option raises an error.
- **Retries.** Rate limits, server errors, dropped connections and truncated
  streams are retried up to three times, with exponential backoff and jitter.
  Authentication errors and other 4xx responses fail immediately.
- **Caching.** Only completed responses are cached, keyed by request body and
  `rollout_id`.

The task model uses a 90-second timeout and the run's disk cache. The reflection
model uses 180 seconds and no cache.

## CLI, outputs and checkpoints

```bash
uv run python -m bird_gepa.cli \
  --prepared data/bird/prepared --databases data/bird/databases \
  --output artifacts/bird-workflow/solver.py
```

| Option | Default |
| --- | --- |
| `--prepared` | `data/bird/prepared` |
| `--databases` | `data/bird/databases`, as `<db_id>/<db_id>.sqlite` |
| `--task-model` | `codex/gpt-6-luna` |
| `--reflection-model` | `codex/gpt-6-astra` |
| `--metric-calls` | 5,500 |
| `--num-threads` | 8 |
| `--reflection-minibatch-size` | 8 |
| `--validation-size` | 300 (use 1376 for the whole split) |
| `--max-predictor-calls` | 10 |
| `--query-penalty` | 0.01 |
| `--seed` | 42, for optimization and the subset sample, not the data split |
| `--output` | `artifacts/bird-workflow/solver.py` |
| `--checkpoint-dir` | `<output directory>/checkpoint` |

The CLI loads the prepared examples without running reference SQL, never loads
the test split, and logs at INFO level. The output directory also holds the model
cache (`lm_cache/`). LM call history is disabled during the run.

GEPA saves checkpoints at iteration boundaries, so rerunning the same command
resumes the candidates and the metric calls already spent. An interrupted
iteration may repeat. Budgets are cumulative, so raise `--metric-calls` to
continue a finished run.

`checkpoint/manifest.json` records everything that affects scores:

- training and validation subset question IDs and inputs
- model IDs
- seed source
- the `SearchSettings`: seed, minibatch size, validation subset size, predictor
  limit and query penalty
- metric version

The metric budget and thread count are left out: the budget is cumulative across
resumes, and threads do not change scores.

A resume with a different setup, or from an older checkpoint without this file,
fails before any model call. Use a new output directory for each experiment.

## Tests

The tests follow the module boundaries: database access, data loading, the seed
in Flex, the metric, optimization, CLI wiring and the Codex model. Tests marked
`slow` run real Flex and GEPA in the Deno sandbox against a mocked Codex
endpoint, including GEPA replacing an editable helper. They are skipped without
Deno.

DSPy 3.3.1 and GEPA 0.1.4 are pinned.

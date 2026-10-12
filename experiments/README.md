# Experiments

The two runs reported in the main [README](../README.md#results). Both used the
default settings. The only difference was the query penalty.

| Folder | Query penalty | Seed → optimized accuracy |
| --- | ---: | --- |
| [query-penalty](query-penalty/) | 0.01 | 62.21% → 67.44% |
| [accuracy-only](accuracy-only/) | 0 | 61.19% → 73.55% |

## Files

Each folder holds:

| File | Contents |
| --- | --- |
| `run.json` | The command, settings and timings |
| `result.json` | Accuracy, query counts and metric calls, before and after |
| `solver.json` | The optimized solver, in DSPy's format |
| `candidates.json` | The source of every candidate GEPA accepted. The first is the seed |
| `candidate_tree.html` | GEPA's search tree. Open it in a browser |
| `run_log.json` | Each iteration's parent, training batch and scores |
| `selection.json` | The training questions, and the 300-question selection subset |
| `run.log` | The full log, including each reflection prompt and proposal |

## Notes

- **These runs used an earlier seed**, which re-ran its final SQL and so made
  four queries per question. The current seed makes three.
- **The query-penalty run was restarted** after Codex `server_is_overloaded`
  errors, which led to the retry handling in `codex.py`. Its `run.log` is
  the log of the final, complete attempt.
- **Paths in `run.json` and the logs point to `artifacts/`**, where the runs
  were written. That directory, the response cache and GEPA's binary state are
  not committed.
- **`solver.json` was saved after the runs**, when the CLI moved to DSPy's
  format. Load it with `load_solver`, as shown in the main README.

## Licence

These files contain questions, evidence, reference SQL and database rows from
[BIRD](https://bird-bench.github.io/), which is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). That licence,
not the repository's MIT licence, applies to that content.

# Data preparation

[download_bird.sh](../scripts/download_bird.sh) fetches the question files from
Hugging Face at pinned revisions, and the database packages from
[BIRD's server](https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip) and
[Google Drive](https://drive.google.com/file/d/13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG/view).
It runs the Hugging Face CLI and gdown through `uvx`, and extracts only the
databases the question files use. Existing files are reused and interrupted
downloads resume, so delete a file to fetch it again.

Both commands default to `data/bird`. To use another directory:

```bash
bash scripts/download_bird.sh /path/to/bird
uv run python -m bird_gepa.prepare --directory /path/to/bird
```

## Steps

[prepare.py](../bird_gepa/prepare.py) reads `raw/` and `databases/`, writes
`prepared/`, and makes no model calls.

1. **Clean.** Trim whitespace around `question` and `evidence`, and turn missing
   evidence into an empty string. SQL is kept as published. Each record gets an
   `example_id`: the source row index for training, the published ID for mini-dev.
2. **Split by database.** Sort the training databases by the SHA-256 of
   `seed:db_id`, which gives a shuffled but repeatable order, and move the
   prefix whose question count is closest to `--validation-fraction` (default
   0.2) into validation. At least one database
   stays in training. Changing `--seed` changes which databases are held out.
3. **Remove duplicates.** Two training records are duplicates when `db_id`,
   `question` and `evidence` match after collapsing whitespace and `casefold()`.
   The first is kept. SQL is not part of the key, and paraphrases are not caught.
4. **Collect reference answers.** Each train and validation reference query runs
   once against the read-only database, with a 600-second deadline, and its full
   result is stored. An empty result is a valid answer. A failing query is logged
   and left out of the examples but stays in the JSON split. Mini-dev queries are
   not run.

## Outputs

| File | Contents |
| --- | --- |
| `train.json`, `validation.json`, `test.json` | Cleaned records for each split |
| `dspy_examples.json` | Training and validation examples: the solver's inputs (`db_id`, `question`, `evidence`) with the reference SQL and its result rows, loaded as `dspy.Example` objects |

BLOB cells are stored as `{"blob_hex": "..."}`, and the examples file is replaced
atomically.

`dspy_examples.json` carries a fingerprint: a hash of the source records, the
split sizes, each database file's size and modification time, the SQLite version
and a format version. The CLI checks it without opening SQLite and refuses stale
examples. Database contents are not rehashed, so rerun preparation after
changing records, database files or SQLite.

# Data preparation

## Sources

| Role | Source | Records |
| --- | --- | --- |
| Train/validation | [birdsql/bird23-train-filtered](https://huggingface.co/datasets/birdsql/bird23-train-filtered) | 6,601 questions across 69 databases |
| Held-out test | [birdsql/bird_mini_dev](https://huggingface.co/datasets/birdsql/bird_mini_dev) SQLite | 500 questions across 11 other databases |

[download_bird.sh](../scripts/download_bird.sh) pins the Hugging Face revisions.
Question files total about 3 MB. The database packages come from
[BIRD's server](https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip) and
[Google Drive](https://drive.google.com/file/d/13VLWIwpw5E3d5DUkMvzw7hvHE67a4XkG/view).
The datasets are CC BY-SA 4.0, so keep the BIRD attribution when sharing data.

## Run

```bash
bash scripts/download_bird.sh
uv run python -m bird_gepa.prepare
```

Downloading needs Bash, uv, jq and unzip. The script runs the
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) and
[gdown](https://github.com/wkentaro/gdown) through `uvx`, so neither has to be
installed separately.

Downloads total about 9.7 GB. Only the nested database ZIP inside the training
archive is kept. Archives take about 10 GB and the extracted databases about 34 GB,
so allow 50 GB of free space. Only databases used by the question files are
extracted. Existing files are reused, and gdown resumes interrupted downloads.
Delete a file to download or extract it again.

Both commands default to `data/bird`. To use another directory, pass it as the
script's first argument and as `--directory`:

```bash
bash scripts/download_bird.sh /path/to/bird
uv run python -m bird_gepa.prepare --directory /path/to/bird
```

Preparation reads `raw/` and `databases/` and writes `prepared/` in that directory.
`--seed` (default 42) and `--validation-fraction` (default 0.2) control the split.

## What preparation does

[prepare.py](../bird_gepa/prepare.py) performs these steps:

1. **Read and clean.** Read the training JSONL and the mini-dev JSON. Trim whitespace
   around `question` and `evidence`, and replace missing evidence with an empty
   string. SQL and other fields are kept unchanged. Add `source_index` and
   `example_id`. Training IDs use the source row index, and mini-dev IDs use the
   published question ID.
2. **Split by database.** Sort training database IDs by the SHA-256 hash of
   `seed:db_id`, and assign to validation the prefix whose question count is
   closest to the validation fraction. At least one database stays in training,
   and no database appears in both splits, so the split is only roughly 80/20.
3. **Remove duplicates.** Training records with the same `db_id`, `question` and
   `evidence` are duplicates. Questions and evidence are compared after
   collapsing whitespace and applying `casefold()`. The first record is kept,
   with its text as stored. SQL is not part of the key, and paraphrases are not
   detected.
4. **Save the splits.** All 500 mini-dev questions form the test split, which
   optimization never loads.
5. **Collect reference answers.** Run each train and validation reference query
   once against its read-only database, with a 600-second deadline, and keep the
   full result rows. Failing queries are logged and left out of the DSPy examples,
   but stay in the JSON splits. Empty results are valid answers. Mini-dev queries
   are not run.
6. **Save DSPy examples.** Only `db_id`, `question` and `evidence` are predictor
   inputs. The reference SQL and answer rows are labels for grading and feedback.
   BLOB values are stored as `{"blob_hex": "..."}`, and the file is replaced
   atomically. Preparation makes no model calls.

## Outputs

With the pinned sources and default settings, `data/bird/prepared/` contains:

| File | Contents |
| --- | --- |
| `train.json` | 5,223 cleaned training records |
| `validation.json` | 1,376 cleaned validation records |
| `test.json` | All 500 mini-dev records |
| `dspy_examples.json` | 5,221 training and 1,376 validation examples with reference answers |

Two duplicate training records are removed, and two training questions whose
reference SQL fails are left out of the examples.

`dspy_examples.json` stores a fingerprint covering:

- a hash of the source records
- the split sizes
- each database file's size and modification time
- the SQLite version
- a format version

The CLI checks this fingerprint without opening SQLite and refuses stale examples.
Database contents are not rehashed. Rerun preparation after changing records,
database files or SQLite version. There are no checksum checks or integrity reports.

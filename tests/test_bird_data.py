import io
import json
import os
import subprocess
import zipfile
from pathlib import Path

from bird_gepa.data import load_examples, locate_databases, source_fingerprint
from bird_gepa.prepare import prepare, split_databases


def test_downloads_questions_and_extracts_both_database_packages(tmp_path):
    train = [{"db_id": "train"}]
    mini = [{"db_id": "mini"}]
    train_records = tmp_path / "train.jsonl"
    mini_records = tmp_path / "mini.json"
    train_records.write_text(json.dumps(train[0]) + "\n")
    mini_records.write_text(json.dumps(mini))
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as archive:
        archive.writestr("train_databases/train/train.sqlite", b"training database")
        archive.writestr("train_databases/train/database_description/items.csv", b"description")
        archive.writestr("train_databases/unused/unused.sqlite", b"unused")
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as archive:
        archive.writestr("train/train_databases.zip", nested.getvalue())
    mini_archive = tmp_path / "mini.zip"
    with zipfile.ZipFile(mini_archive, "w") as archive:
        archive.writestr("minidev/MINIDEV/dev_databases/mini/mini.sqlite", b"mini database")
    tools = tmp_path / "tools"
    tools.mkdir()
    uvx = tools / "uvx"
    uvx.write_text("""#!/usr/bin/env bash
printf '%s\n' "$1" >> "$BIRD_TEST_LOG"
if [[ $1 == hf ]]; then
    if [[ $3 == *bird23-train-filtered* ]]; then
        printf '%s\n' "$BIRD_TEST_TRAIN"
    else
        printf '%s\n' "$BIRD_TEST_MINI"
    fi
elif [[ $2 == https:* ]]; then
    cp "$BIRD_TEST_OUTER" "$4"
else
    cp "$BIRD_TEST_ARCHIVE" "$4"
fi
""")
    uvx.chmod(0o755)
    log = tmp_path / "calls.log"
    environment = dict(
        os.environ,
        PATH=str(tools) + os.pathsep + os.environ["PATH"],
        BIRD_TEST_TRAIN=str(train_records),
        BIRD_TEST_MINI=str(mini_records),
        BIRD_TEST_OUTER=str(outer),
        BIRD_TEST_ARCHIVE=str(mini_archive),
        BIRD_TEST_LOG=str(log),
    )
    directory = tmp_path / "bird data"
    script = Path(__file__).resolve().parents[1] / "scripts/download_bird.sh"
    subprocess.run(["bash", str(script), str(directory)], env=environment, check=True)
    assert log.read_text().splitlines() == ["hf", "hf", "gdown", "gdown"]
    assert (directory / "raw/train_filtered.jsonl").read_text() == train_records.read_text()
    assert (directory / "raw/mini_dev_sqlite.json").read_text() == mini_records.read_text()
    assert (directory / "databases/train/train.sqlite").read_bytes() == b"training database"
    assert (directory / "databases/mini/mini.sqlite").read_bytes() == b"mini database"
    assert sorted(
        str(path.relative_to(directory / "databases"))
        for path in (directory / "databases").rglob("*")
        if path.is_file()
    ) == ["mini/mini.sqlite", "train/train.sqlite"]
    subprocess.run(["bash", str(script), str(directory)], env=environment, check=True)
    assert log.read_text().splitlines() == ["hf", "hf", "gdown", "gdown"]


def test_prepare_splits_databases_and_preserves_test_records_and_sql(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    train = [
        {
            "db_id": f"db_{index % 10}",
            "question": f" Question {index}? ",
            "evidence": None,
            "SQL": "SELECT ' a  b\n '  ",
        }
        for index in range(100)
    ]
    train.append(dict(train[0], question="question  0?"))
    mini = [{"db_id": "mini", "question": "Test?", "SQL": "SELECT 1", "question_id": 42}]
    (raw / "train_filtered.jsonl").write_text("\n".join(json.dumps(row) for row in train))
    (raw / "mini_dev_sqlite.json").write_text(json.dumps(mini))
    for index in range(10):
        path = tmp_path / "databases" / f"db_{index}" / f"db_{index}.sqlite"
        path.parent.mkdir(parents=True)
        path.touch()
    counts = prepare(tmp_path)
    output = tmp_path / "prepared"
    splits = {name: json.loads((output / f"{name}.json").read_text()) for name in counts}
    assert counts == {"train": 80, "validation": 20, "test": 1}
    assert not {row["db_id"] for row in splits["train"]} & {
        row["db_id"] for row in splits["validation"]
    }
    rows = splits["train"] + splits["validation"]
    assert {row["example_id"] for row in rows} == {
        f"train-filtered:{index}" for index in range(100)
    }
    assert all(row["SQL"] == train[0]["SQL"] and row["evidence"] == "" for row in rows)
    assert splits["test"] == [dict(mini[0], evidence="", example_id="mini-dev:42", source_index=0)]
    assert {path.name for path in output.iterdir()} == {
        "train.json",
        "validation.json",
        "test.json",
        "dspy_examples.json",
    }
    databases = locate_databases(splits["train"] + splits["validation"], tmp_path / "databases")
    train_examples, validation_examples = load_examples(
        output / "dspy_examples.json",
        source_fingerprint(splits["train"], splits["validation"], databases),
    )
    assert (len(train_examples), len(validation_examples)) == (80, 20)
    assert all(
        example.expected_rows == [(" a  b\n ",)] for example in train_examples + validation_examples
    )
    assert split_databases(train, 42, 0.2) == split_databases(list(reversed(train)), 42, 0.2)

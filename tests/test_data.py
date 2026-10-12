import json
import sqlite3
from collections import Counter
from contextlib import closing

import dspy
import pytest

from bird_gepa.data import build_examples, locate_databases, read_splits, sample_by_database


def test_grade_uses_full_results_and_inputs_exclude_labels(tmp_path, demo):
    rows = demo
    database = tmp_path / "demo/demo.sqlite"
    with closing(sqlite3.connect(database)) as conn:
        conn.executemany("INSERT INTO numbers VALUES (?)", [(n,) for n in range(11, 41)])
        conn.commit()
    rows[0]["SQL"] = "SELECT value FROM numbers"
    databases = locate_databases(rows, tmp_path)
    examples = build_examples(rows, databases)
    assert len(examples[0].expected_rows) == 40
    assert "reference_sql" not in examples[0].inputs()
    assert "expected_rows" not in examples[0].inputs()
    assert "example_id" not in examples[0].inputs()
    assert set(examples[0].inputs()) == {"db_id", "question", "evidence"}


def test_query_does_not_create_missing_database(tmp_path, demo):
    databases = locate_databases(demo + [{"db_id": "missing"}], tmp_path)
    (tmp_path / "missing").mkdir()
    with pytest.raises(sqlite3.OperationalError, match="unable to open"):
        databases["missing"].query("SELECT 1")
    assert not (tmp_path / "missing/missing.sqlite").exists()


def test_failed_reference_is_skipped_when_building_examples(tmp_path, demo, caplog):
    demo[0]["SQL"] = "SELECT * FROM missing"
    databases = locate_databases(demo, tmp_path)
    examples = build_examples(demo, databases)
    assert len(examples) == len(demo) - 1
    assert [example.question for example in examples] == [row["question"] for row in demo[1:]]
    assert "Skipping" in caplog.text
    assert databases["demo"].query("SELECT 1") == ([(1,)], None)


def test_prepared_rows_preserve_complete_splits_without_test_file(tmp_path):
    splits = {}
    for split, size in [("train", 20), ("validation", 10)]:
        rows = [{"db_id": split, "example_id": f"{split}:{n}"} for n in range(size)]
        splits[split] = rows
        (tmp_path / f"{split}.json").write_text(json.dumps(rows))
    train, validation = read_splits(tmp_path)
    assert train == splits["train"]
    assert validation == splits["validation"]


def test_validation_sample_is_fixed_and_covers_databases():
    examples = [
        dspy.Example(db_id=db_id, example_id=f"{db_id}:{index}")
        for db_id, size in [("large", 700), ("medium", 150), ("small", 20), ("tiny", 3)]
        for index in range(size)
    ]
    original_ids = [example.example_id for example in examples]
    selected = sample_by_database(examples, size=300, seed=42)
    selected_ids = [example.example_id for example in selected]
    assert len(selected_ids) == len(set(selected_ids)) == 300
    assert Counter(example.db_id for example in selected) == {
        "large": 241,
        "medium": 51,
        "small": 7,
        "tiny": 1,
    }
    assert selected_ids == [
        example.example_id for example in sample_by_database(examples, size=300, seed=42)
    ]
    assert selected_ids != [
        example.example_id for example in sample_by_database(examples, size=300, seed=43)
    ]
    assert [example.example_id for example in examples] == original_ids
    assert set(selected_ids) <= set(original_ids)


def test_validation_sample_keeps_small_datasets_complete():
    examples = [dspy.Example(db_id="small", example_id=str(index)) for index in range(4)]
    assert sample_by_database(examples, size=300, seed=42) == examples


def test_validation_sample_rounds_proportional_counts_to_requested_size():
    examples = [dspy.Example(db_id=str(index // 10)) for index in range(30)]
    selected = sample_by_database(examples, size=14, seed=42)
    assert Counter(example.db_id for example in selected) == {"0": 5, "1": 5, "2": 4}
    selected = sample_by_database(examples, size=2, seed=42)
    assert Counter(example.db_id for example in selected) == {"0": 1, "1": 1}

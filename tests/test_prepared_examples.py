import json
import os
import shutil
import sqlite3

import pytest

from bird_gepa.data import load_examples, read_sources, save_examples
from bird_gepa.prepare import prepare_examples


@pytest.fixture
def prepared(tmp_path, demo):
    (tmp_path / "train.json").write_text(json.dumps(demo[:4]))
    validation = [dict(row, db_id="validation") for row in demo[4:]]
    (tmp_path / "validation.json").write_text(json.dumps(validation))
    (tmp_path / "validation").mkdir()
    shutil.copyfile(tmp_path / "demo/demo.sqlite", tmp_path / "validation/validation.sqlite")
    return tmp_path


def fingerprint(prepared):
    return read_sources(prepared, prepared).fingerprint


def test_preparation_preserves_full_results_types_and_input_boundaries(prepared, monkeypatch):
    path = prepared / "train.json"
    records = json.loads(path.read_text())
    records[0]["SQL"] = (
        "SELECT value, NULL, 1.25, 'é', x'00ff' FROM numbers "
        "CROSS JOIN (SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3)"
    )
    records[1]["SQL"] = "SELECT value FROM numbers WHERE 0"
    path.write_text(json.dumps(records))
    # No test split is created or needed.
    output = prepare_examples(prepared, prepared)

    def unexpected_connection(*args, **kwargs):
        pytest.fail("Restoring prepared examples must not open SQLite")

    monkeypatch.setattr(sqlite3, "connect", unexpected_connection)
    train, validation = load_examples(output, fingerprint(prepared))
    assert (len(train), len(validation)) == (4, 4)
    assert len(train[0].expected_rows) == 30
    assert train[0].expected_rows[0] == (1, None, 1.25, "é", b"\x00\xff")
    assert len(set(train[0].expected_rows)) == 10
    assert train[1].expected_rows == []
    assert train[0].reference_sql == records[0]["SQL"]
    assert [example.question for example in train] == [row["question"] for row in records]
    for example in train + validation:
        assert set(example.inputs()) == {"db_id", "question", "evidence"}
        assert "reference_sql" in example.labels()
        assert "expected_rows" in example.labels()


@pytest.mark.parametrize("changed", ["records", "database", "sqlite_version"])
def test_changed_sources_require_preparation(prepared, monkeypatch, changed):
    output = prepare_examples(prepared, prepared)
    if changed == "records":
        path = prepared / "train.json"
        records = json.loads(path.read_text())
        records.reverse()
        path.write_text(json.dumps(records))
    elif changed == "database":
        path = prepared / "demo/demo.sqlite"
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    else:
        monkeypatch.setattr(sqlite3, "sqlite_version", "different")
    with pytest.raises(ValueError, match="stale"):
        load_examples(output, fingerprint(prepared))


def test_missing_examples_are_rejected(prepared):
    output = prepared / "dspy_examples.json"
    with pytest.raises(FileNotFoundError):
        load_examples(output, fingerprint(prepared))


def test_preparation_saves_examples_with_failed_references_omitted(prepared, caplog):
    path = prepared / "train.json"
    records = json.loads(path.read_text())
    records[0]["SQL"] = "SELECT missing FROM numbers"
    path.write_text(json.dumps(records))
    output = prepare_examples(prepared, prepared)
    train, validation = load_examples(output, fingerprint(prepared))
    assert (len(train), len(validation)) == (3, 4)
    assert [example.question for example in train] == [row["question"] for row in records[1:]]
    assert "Skipping" in caplog.text


def test_failed_serialization_preserves_previous_artifact(prepared):
    output = prepare_examples(prepared, prepared)
    previous = output.read_bytes()
    source = fingerprint(prepared)
    train, validation = load_examples(output, source)
    train[0].expected_rows = [(object(),)]
    with pytest.raises(TypeError, match="Cannot serialize"):
        save_examples(output, train, validation, source)
    assert output.read_bytes() == previous
    assert not output.with_suffix(".json.tmp").exists()

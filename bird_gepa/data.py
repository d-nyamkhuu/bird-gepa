"""Load BIRD examples and database paths, preserving split boundaries."""

import hashlib
import json
import logging
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import dspy

from bird_gepa.database import Database, decode_blob, encode_blob

INPUT_FIELDS = ("db_id", "question", "evidence")
EXAMPLES_FILE = "dspy_examples.json"
logger = logging.getLogger(__name__)


class BirdRecord(TypedDict):
    """A cleaned BIRD question, as saved in the prepared splits. Other source keys are kept."""

    example_id: str
    db_id: str
    question: str
    evidence: str
    SQL: str


def locate_databases(records: list[BirdRecord], directory: Path) -> dict[str, Database]:
    """Locate each database without keeping connections open."""
    return {
        db_id: Database(directory / db_id / f"{db_id}.sqlite")
        for db_id in dict.fromkeys(record["db_id"] for record in records)
    }


def build_examples(
    records: list[BirdRecord], databases: dict[str, Database], *, reference_timeout: float = 600
) -> list[dspy.Example]:
    """Build examples by running each reference query against its database."""
    examples = []
    for record in records:
        expected_rows, error = databases[record["db_id"]].query(
            record["SQL"], timeout=reference_timeout
        )
        if error:
            logger.warning("Skipping %s: %s", record["example_id"], error)
            continue
        examples.append(
            dspy.Example(
                example_id=record["example_id"],
                db_id=record["db_id"],
                question=record["question"],
                evidence=record["evidence"],
                reference_sql=record["SQL"],
                expected_rows=expected_rows,
            ).with_inputs(*INPUT_FIELDS)
        )
    return examples


def read_splits(directory: Path) -> tuple[list[BirdRecord], list[BirdRecord]]:
    """Load complete frozen training/validation splits without opening the test set."""
    train = json.loads((directory / "train.json").read_text())
    validation = json.loads((directory / "validation.json").read_text())
    return train, validation


def sample_by_database(examples: list[dspy.Example], *, size: int, seed: int) -> list[dspy.Example]:
    """Select a fixed subset proportionally to each database's question count."""
    if size >= len(examples):
        return list(examples)

    groups: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        groups.setdefault(example.db_id, []).append(index)
    counts = {}
    remainders = {}
    for db_id, indices in groups.items():
        count, remainder = divmod(size * len(indices), len(examples))
        counts[db_id] = count
        remainders[db_id] = remainder
    ordered = sorted(groups, key=lambda db_id: (-remainders[db_id], db_id))
    for db_id in ordered[: size - sum(counts.values())]:
        counts[db_id] += 1

    rng = random.Random(seed)
    selected = set()
    for db_id in sorted(groups):
        selected.update(rng.sample(groups[db_id], counts[db_id]))
    return [example for index, example in enumerate(examples) if index in selected]


def source_fingerprint(
    train: list[BirdRecord], validation: list[BirdRecord], databases: dict[str, Database]
) -> dict:
    """Identify source records and database files without executing SQL."""
    files = {}
    for db_id, database in databases.items():
        stat = database.path.stat()
        files[db_id] = [stat.st_size, stat.st_mtime_ns]
    records = json.dumps((train, validation), sort_keys=True).encode()
    return {
        "version": 1,
        "records_sha256": hashlib.sha256(records).hexdigest(),
        "counts": [len(train), len(validation)],
        "databases": files,
        "sqlite_version": sqlite3.sqlite_version,
    }


@dataclass(frozen=True)
class Sources:
    """Frozen split records, the databases they refer to, and a fingerprint of both."""

    train: list[BirdRecord]
    validation: list[BirdRecord]
    databases: dict[str, Database]
    fingerprint: dict


def read_sources(prepared_dir: Path, databases_dir: Path) -> Sources:
    """Read the frozen splits and locate their databases, without executing SQL."""
    train, validation = read_splits(prepared_dir)
    databases = locate_databases(train + validation, databases_dir)
    fingerprint = source_fingerprint(train, validation, databases)
    return Sources(train, validation, databases, fingerprint)


def save_examples(
    path: Path,
    train: list[dspy.Example],
    validation: list[dspy.Example],
    fingerprint: dict,
) -> None:
    """Publish complete examples atomically, preserving SQLite BLOB values."""
    payload = {
        "fingerprint": fingerprint,
        "train": [dict(example.items()) for example in train],
        "validation": [dict(example.items()) for example in validation],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, default=encode_blob)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_examples(path: Path, fingerprint: dict) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Restore prepared examples and their input fields without running reference SQL."""
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream, object_hook=decode_blob)
    if payload["fingerprint"] != fingerprint:
        raise ValueError("Prepared examples are stale. Rerun python -m bird_gepa.prepare.")
    for record in payload["train"] + payload["validation"]:
        record["expected_rows"] = [tuple(row) for row in record["expected_rows"]]
    train, validation = (
        [dspy.Example(**record).with_inputs(*INPUT_FIELDS) for record in payload[name]]
        for name in ("train", "validation")
    )
    return train, validation


@dataclass(frozen=True)
class PreparedData:
    """Training and validation examples, with the databases they refer to."""

    train: list[dspy.Example]
    validation: list[dspy.Example]
    databases: dict[str, Database]

    @classmethod
    def load(cls, prepared_dir: Path, databases_dir: Path) -> "PreparedData":
        """Load saved examples, checking they still match their source records and files."""
        sources = read_sources(prepared_dir, databases_dir)
        train, validation = load_examples(prepared_dir / EXAMPLES_FILE, sources.fingerprint)
        return cls(train=train, validation=validation, databases=sources.databases)

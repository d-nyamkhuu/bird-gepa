"""Clean and split BIRD records, then save examples with full reference answers."""

import argparse
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path

from bird_gepa.data import (
    EXAMPLES_FILE,
    build_examples,
    locate_databases,
    read_splits,
    save_examples,
    source_fingerprint,
)

logger = logging.getLogger(__name__)


def prepare_examples(prepared_dir: Path, databases_dir: Path) -> Path:
    train_records, validation_records = read_splits(prepared_dir)
    databases = locate_databases(train_records + validation_records, databases_dir)
    fingerprint = source_fingerprint(train_records, validation_records, databases)
    examples = []
    for name, records in (("train", train_records), ("validation", validation_records)):
        logger.info("Preparing %s: %d reference queries", name, len(records))
        examples.append(build_examples(records, databases))
        logger.info("Prepared %s: %d examples", name, len(examples[-1]))
    train, validation = examples
    output = prepared_dir / EXAMPLES_FILE
    save_examples(output, train, validation, fingerprint)
    logger.info("Saved %s", output)
    return output


def read_records(path: Path, source: str) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    rows = (
        json.loads(text)
        if path.suffix == ".json"
        else [json.loads(line) for line in text.splitlines()]
    )
    return [
        dict(
            row,
            question=row["question"].strip(),
            evidence=(row.get("evidence") or "").strip(),
            example_id=f"{source}:{row.get('question_id', index)}",
            source_index=index,
        )
        for index, row in enumerate(rows)
    ]


def dedupe_key(row: dict) -> tuple[str, str, str]:
    """Treat questions as duplicates when they differ only in case or whitespace."""
    question, evidence = (" ".join(row[k].split()).casefold() for k in ("question", "evidence"))
    return row["db_id"], question, evidence


def split_databases(rows: list[dict], seed: int, validation_fraction: float) -> dict[str, str]:
    counts = Counter(row["db_id"] for row in rows)
    ordered = sorted(counts, key=lambda db: hashlib.sha256(f"{seed}:{db}".encode()).hexdigest())
    target = len(rows) * validation_fraction
    total, candidates = 0, []
    for index, db in enumerate(ordered[:-1], 1):
        total += counts[db]
        candidates.append((abs(total - target), index))
    _, boundary = min(candidates)
    return {db: "validation" if db in ordered[:boundary] else "train" for db in ordered}


def prepare(root: Path, *, seed: int = 42, validation_fraction: float = 0.2) -> dict:
    raw, output = root / "raw", root / "prepared"
    train = read_records(raw / "train_filtered.jsonl", "train-filtered")
    test = read_records(raw / "mini_dev_sqlite.json", "mini-dev")
    assignments = split_databases(train, seed, validation_fraction)
    unique: dict[tuple, dict] = {}
    for row in train:
        unique.setdefault(dedupe_key(row), row)
    splits = {
        "train": [row for row in unique.values() if assignments[row["db_id"]] == "train"],
        "validation": [row for row in unique.values() if assignments[row["db_id"]] == "validation"],
        "test": test,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        (output / f"{name}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    prepare_examples(output, root / "databases")
    return {name: len(rows) for name, rows in splits.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/bird"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    counts = prepare(args.directory, seed=args.seed, validation_fraction=args.validation_fraction)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()

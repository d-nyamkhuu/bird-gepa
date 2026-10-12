"""Run BIRD SQL optimization with Codex, DSPy Flex, and GEPA."""

import argparse
import logging
from pathlib import Path

import dspy

from bird_gepa.codex import CodexLM
from bird_gepa.data import PreparedData
from bird_gepa.optimization import SearchSettings, optimize

logger = logging.getLogger(__name__)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return number


def fraction(value: str) -> float:
    number = float(value)
    # Written this way round so that nan is rejected too.
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("Use a number between 0 and 1.")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = SearchSettings()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("data/bird/prepared"),
        help="Prepared BIRD directory with frozen splits",
    )
    parser.add_argument(
        "--databases",
        type=Path,
        default=Path("data/bird/databases"),
        help="Directory containing <db_id>/<db_id>.sqlite",
    )
    parser.add_argument("--metric-calls", type=positive_int, default=5500)
    parser.add_argument(
        "--num-threads", type=positive_int, default=8, help="Parallel evaluation workers"
    )
    parser.add_argument(
        "--reflection-minibatch-size",
        type=positive_int,
        default=defaults.reflection_minibatch_size,
    )
    parser.add_argument(
        "--max-predictor-calls",
        type=positive_int,
        default=defaults.max_predictor_calls,
        help="Maximum bridged Flex predictor calls per question",
    )
    parser.add_argument(
        "--query-penalty",
        type=fraction,
        default=defaults.query_penalty,
        help="Correct-answer score penalty for each solver query after the first (0..1)",
    )
    parser.add_argument(
        "--validation-size",
        type=positive_int,
        default=defaults.validation_size,
        help="Fixed GEPA validation subset, sampled proportionally by database",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the full validation evaluation before optimization",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, help="Defaults to <output directory>/checkpoint"
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--task-model", default="codex/gpt-6-luna")
    parser.add_argument("--reflection-model", default="codex/gpt-6-astra")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/bird-solver/solver.json"),
        help="Optimized solver, in DSPy's JSON format",
    )
    args = parser.parse_args(argv)
    if args.output.suffix != ".json":
        parser.error("--output must be a .json file.")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.output.parent / "checkpoint"
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    dspy.configure_cache(
        disk_cache_dir=str(args.output.parent / "lm_cache"), memory_max_entries=2000
    )
    # Release the connection opened by configure_cache. CodexLM opens one per call.
    dspy.cache.disk_cache.close()
    data = PreparedData.load(args.prepared, args.databases)
    settings = SearchSettings(
        seed=args.seed,
        reflection_minibatch_size=args.reflection_minibatch_size,
        validation_size=args.validation_size,
        max_predictor_calls=args.max_predictor_calls,
        query_penalty=args.query_penalty,
    )
    result = optimize(
        data.train,
        data.validation,
        data.databases,
        task_lm=CodexLM(args.task_model, timeout=90, cache=True),
        reflection_lm=CodexLM(args.reflection_model, timeout=180, cache=False),
        settings=settings,
        max_metric_calls=args.metric_calls,
        num_threads=args.num_threads,
        skip_baseline=args.skip_baseline,
        checkpoint_dir=args.checkpoint_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Save the solver without its Codex model. load_solver attaches a model on load.
    result.program.reset()
    result.program.save(args.output)
    logger.info("Solver: %s", args.output)


if __name__ == "__main__":
    main()

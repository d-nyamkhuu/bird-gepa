"""Evaluate the seed solver, and search for a better one with GEPA."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import dspy
from dspy.evaluate import EvaluationResult

from bird_gepa.data import sample_by_database
from bird_gepa.database import Database, QueryTrace
from bird_gepa.metric import METRIC_VERSION, ExecutionMetric
from bird_gepa.program import SEED_SOURCE, build_program

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class SearchSettings:
    """Everything that changes GEPA scores. Recorded in the checkpoint manifest."""

    seed: int = 42
    reflection_minibatch_size: int = 8
    validation_size: int = 300
    max_predictor_calls: int = 10
    query_penalty: float = 0.01


def checkpoint_manifest(
    train: list[dspy.Example],
    validation_subset: list[dspy.Example],
    settings: SearchSettings,
    *,
    task_lm: dspy.BaseLM,
    reflection_lm: dspy.BaseLM,
) -> dict:
    """Describe a run, so that a checkpoint is only resumed under the same setup.

    The metric budget is cumulative across resumes, and thread count does not change
    scores, so neither is recorded.
    """

    def describe(examples: list[dspy.Example]) -> list[dict]:
        return [{"example_id": e.get("example_id"), **dict(e.inputs())} for e in examples]

    return {
        "train": describe(train),
        "validation": describe(validation_subset),
        **asdict(settings),
        "metric_version": METRIC_VERSION,
        "task_model": task_lm.model,
        "reflection_model": reflection_lm.model,
        "solver_source": SEED_SOURCE,
    }


def check_checkpoint(checkpoint_dir: Path, manifest: dict) -> None:
    """Record the manifest on first use, and refuse to resume under a different one."""
    path = checkpoint_dir / MANIFEST_FILE
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Checkpoint manifest has changed. Use a new --checkpoint-dir.")
        return
    if (checkpoint_dir / "gepa_state.bin").exists():
        raise ValueError("Checkpoint has no manifest. Use a new --checkpoint-dir.")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


@dataclass(frozen=True)
class EvaluationSummary:
    """Execution accuracy alongside GEPA's penalized score, over one full evaluation."""

    total: int
    correct: int
    score: float
    queries: int
    query_seconds: float

    @property
    def accuracy(self) -> float:
        """Execution accuracy, in percent."""
        return 100.0 * self.correct / self.total


def summarize(result: EvaluationResult) -> EvaluationSummary:
    # `dspy.Evaluate` substitutes an empty prediction and a 0.0 score for examples that raised.
    traces = [
        getattr(prediction, "query_trace", QueryTrace()) for _, prediction, _ in result.results
    ]
    return EvaluationSummary(
        total=len(result.results),
        correct=sum(bool(getattr(score, "correct", False)) for _, _, score in result.results),
        score=sum(float(score) for _, _, score in result.results) / len(result.results),
        queries=sum(trace.count for trace in traces),
        query_seconds=sum(trace.seconds for trace in traces),
    )


def log_summary(label: str, summary: EvaluationSummary) -> None:
    logger.info(
        "%s full validation: EX %.2f%% (%d/%d); GEPA score %.4f; "
        "solver queries %d total, %.2f/question; query tool time %.3fs/question",
        label,
        summary.accuracy,
        summary.correct,
        summary.total,
        summary.score,
        summary.queries,
        summary.queries / summary.total,
        summary.query_seconds / summary.total,
    )


@dataclass(frozen=True)
class OptimizationResult:
    program: dspy.Flex
    baseline: EvaluationSummary | None
    optimized: EvaluationSummary


def optimize(
    train: list[dspy.Example],
    validation: list[dspy.Example],
    databases: dict[str, Database],
    *,
    task_lm: dspy.BaseLM,
    reflection_lm: dspy.BaseLM,
    settings: SearchSettings,
    max_metric_calls: int,
    num_threads: int = 8,
    skip_baseline: bool = False,
    checkpoint_dir: Path | None = None,
) -> OptimizationResult:
    """Evaluate the seed solver, search for a better one, and evaluate that.

    Both evaluations use the full validation split. GEPA selects on a fixed subset of it.
    """
    metric = ExecutionMetric(databases, query_penalty=settings.query_penalty)
    validation_subset = sample_by_database(
        validation, size=settings.validation_size, seed=settings.seed
    )
    if not train or not validation_subset:
        raise ValueError("Training and validation splits must both contain examples.")
    if max_metric_calls <= len(validation_subset):
        raise ValueError("Use a metric budget larger than the validation split.")
    if checkpoint_dir is not None:
        manifest = checkpoint_manifest(
            train, validation_subset, settings, task_lm=task_lm, reflection_lm=reflection_lm
        )
        check_checkpoint(checkpoint_dir, manifest)
    logger.info(
        "GEPA: %d training examples, %d/%d validation examples across %d databases; "
        "%d examples per reflection; at most %d predictor calls/question; "
        "query penalty %g after the first solver query",
        len(train),
        len(validation_subset),
        len(validation),
        len({example.db_id for example in validation_subset}),
        settings.reflection_minibatch_size,
        settings.max_predictor_calls,
        settings.query_penalty,
    )
    program = build_program(databases, task_lm, settings.max_predictor_calls)
    # Per-call history is never read, and would otherwise grow with every metric call.
    with dspy.context(lm=task_lm, adapter=dspy.JSONAdapter(), disable_history=True):
        evaluate = dspy.Evaluate(
            devset=validation,
            metric=metric,
            num_threads=num_threads,
            max_errors=1,
            provide_traceback=True,
        )
        baseline = None
        if not skip_baseline:
            baseline = summarize(evaluate(program))
            log_summary("Baseline", baseline)
        optimized_program = dspy.GEPA(
            metric=metric,
            reflection_lm=reflection_lm,
            max_metric_calls=max_metric_calls,
            reflection_minibatch_size=settings.reflection_minibatch_size,
            num_threads=num_threads,
            use_merge=False,
            seed=settings.seed,
            log_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
            track_stats=True,
        ).compile(program, trainset=train, valset=validation_subset)
        optimized = summarize(evaluate(optimized_program))
        log_summary("Optimized", optimized)
    if baseline is None:
        logger.info("Development EX: %.1f%% (baseline evaluation skipped)", optimized.accuracy)
    else:
        logger.info("Development EX: %.1f%% -> %.1f%%", baseline.accuracy, optimized.accuracy)
    return OptimizationResult(optimized_program, baseline, optimized)

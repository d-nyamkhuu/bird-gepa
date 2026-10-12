import json
import re
import shutil

import dspy
import httpx
import pytest
from dspy.evaluate import EvaluationResult
from gepa.core.state import GEPAState

from bird_gepa.codex import CodexLM
from bird_gepa.data import build_examples, locate_databases
from bird_gepa.database import QueryRecord, QueryTrace
from bird_gepa.metric import METRIC_VERSION
from bird_gepa.optimization import MANIFEST_FILE, SearchSettings, check_checkpoint, optimize

REVISED_SOURCE = """
class SQLSolver(dspy.Module):
    def __init__(self):
        super().__init__()

    def execute_sql(self, db_id, threshold):
        result = query_database(
            db_id, "SELECT COUNT(*) FROM numbers WHERE value > ?",
            parameters=[threshold], row_limit=1
        )
        if result["error"]:
            raise ValueError(result["error"])
        assert result["rows"] == [[10 - threshold]]
        return "SELECT COUNT(*) FROM numbers WHERE value > " + str(threshold)

    def forward(self, **inputs):
        threshold = int(inputs["question"].split()[-1].rstrip("?"))
        sql = self.execute_sql(inputs["db_id"], threshold)
        return dspy.Prediction(sql=sql)
"""


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("deno") is None, reason="Flex integration requires Deno")
@pytest.mark.parametrize(("num_threads", "seed_correct"), [(1, False), (4, False), (4, True)])
def test_flex_gepa_with_mocked_codex(
    tmp_path, demo, codex_login, codex_transport, num_threads, seed_correct
):
    auth = codex_login()
    models = []

    def handler(request):
        body = json.loads(request.content)
        models.append(body["model"])
        if body["model"] == "reflection":
            prompt = str(body["input"])
            assert "Solver queries: 3" in prompt
            assert "Predictor trace calls: 2" in prompt
            if not seed_correct:
                assert "Missing distinct rows:" in prompt
            assert "Host query trace" in prompt
        sql = "SELECT 0"
        if seed_correct and body["model"] == "task":
            match = re.search(r"How many numbers are greater than (\d+)\?", str(body["input"]))
            assert match is not None
            sql = "SELECT COUNT(*) FROM numbers WHERE value > " + match.group(1)
        content = (
            {"revised_source": REVISED_SOURCE}
            if body["model"] == "reflection"
            else {"reasoning": "Synthetic SQL for this test.", "sql": sql}
        )
        event = {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": json.dumps(content)}],
                    }
                ],
            },
        }
        return httpx.Response(200, text=f"data: {json.dumps(event)}\n\n")

    codex_transport(handler)
    databases = locate_databases(demo, tmp_path)
    examples = build_examples(demo, databases)
    result = optimize(
        examples[:4],
        examples[4:],
        databases,
        task_lm=CodexLM("codex/task", auth_file=auth, cache=False),
        reflection_lm=CodexLM("codex/reflection", auth_file=auth, cache=False),
        settings=SearchSettings(),
        max_metric_calls=12,
        num_threads=num_threads,
        checkpoint_dir=tmp_path / "checkpoint",
    )
    assert result.baseline is not None
    assert result.baseline.accuracy == (100.0 if seed_correct else 0.0)
    assert result.optimized.accuracy == 100.0
    assert "reflection" in models and "task" in models
    assert result.program.module_src.strip() == REVISED_SOURCE.strip()
    assert json.loads(auth.read_text())["tokens"]["access_token"] not in result.program.module_src
    state = GEPAState.load(str(tmp_path / "checkpoint"))
    assert len(state.program_candidates) == 2
    assert state.total_num_evals >= 12
    assert state.program_full_scores_val_set == pytest.approx([0.98 if seed_correct else 0, 1])


@pytest.mark.parametrize("skip_baseline", [False, True])
def test_subset_guides_search_and_full_validation_is_reported(
    tmp_path, monkeypatch, caplog, skip_baseline
):
    train = [
        dspy.Example(db_id="train", question="train", evidence="").with_inputs(
            "db_id", "question", "evidence"
        )
    ]
    validation = [
        dspy.Example(
            db_id=f"validation-{index % 15}", question=str(index), evidence=""
        ).with_inputs("db_id", "question", "evidence")
        for index in range(400)
    ]
    record = QueryRecord("demo", "SELECT 1", None, 1, False, None, 0.005)
    evaluations = []
    compilations = []
    caplog.set_level("INFO", logger="bird_gepa.optimization")

    def evaluate(**kwargs):
        assert kwargs["devset"] == validation

        def score(solver):
            evaluations.append(solver)
            assert solver._max_predictor_calls == 10
            return EvaluationResult(
                score=48.5,
                results=[
                    (
                        example,
                        dspy.Prediction(query_trace=QueryTrace([record] * 4)),
                        dspy.Prediction(
                            correct=index % 2 == 0, score=0.97 if index % 2 == 0 else 0
                        ),
                    )
                    for index, example in enumerate(validation)
                ],
            )

        return score

    class Optimizer:
        def __init__(self, **kwargs):
            assert kwargs["reflection_minibatch_size"] == 8

        def compile(self, solver, *, trainset, valset):
            assert len(evaluations) == (0 if skip_baseline else 1)
            compilations.append(valset)
            assert trainset == train
            assert len(valset) == 300
            assert {example.db_id for example in valset} == {
                example.db_id for example in validation
            }
            return solver

    monkeypatch.setattr("bird_gepa.optimization.dspy.Evaluate", evaluate)
    monkeypatch.setattr("bird_gepa.optimization.dspy.GEPA", Optimizer)
    result = optimize(
        train,
        validation,
        {},
        task_lm=dspy.LM("test/task"),
        reflection_lm=dspy.LM("test/reflection"),
        settings=SearchSettings(),
        max_metric_calls=5500,
        skip_baseline=skip_baseline,
        checkpoint_dir=tmp_path / "checkpoint",
    )
    assert result.baseline == (None if skip_baseline else result.optimized)
    assert result.optimized.accuracy == 50.0
    assert len(evaluations) == (1 if skip_baseline else 2)
    assert len(compilations) == 1
    manifest = json.loads((tmp_path / "checkpoint" / MANIFEST_FILE).read_text())
    assert len(manifest["validation"]) == 300
    assert manifest["train"] == [{"example_id": None, **dict(train[0].inputs())}]
    assert manifest["max_predictor_calls"] == 10
    assert manifest["metric_version"] == METRIC_VERSION
    assert manifest["query_penalty"] == 0.01
    assert manifest["validation_size"] == 300
    assert "max_metric_calls" not in manifest
    assert "EX 50.00% (200/400); GEPA score 0.4850" in caplog.text
    assert "solver queries 1600 total, 4.00/question" in caplog.text
    for changed in [SearchSettings(query_penalty=0.02), SearchSettings(max_predictor_calls=5)]:
        with pytest.raises(ValueError, match="manifest has changed"):
            optimize(
                train,
                validation,
                {},
                task_lm=dspy.LM("test/task"),
                reflection_lm=dspy.LM("test/reflection"),
                settings=changed,
                max_metric_calls=5500,
                skip_baseline=skip_baseline,
                checkpoint_dir=tmp_path / "checkpoint",
            )
    assert len(evaluations) == (1 if skip_baseline else 2)


def test_checkpoint_requires_unchanged_manifest(tmp_path):
    manifest = {"validation": ["one", "two"], "reflection_minibatch_size": 8}
    check_checkpoint(tmp_path, manifest)
    check_checkpoint(tmp_path, manifest)
    with pytest.raises(ValueError, match="manifest has changed"):
        check_checkpoint(tmp_path, {**manifest, "validation": ["two", "one"]})
    assert json.loads((tmp_path / MANIFEST_FILE).read_text()) == manifest


def test_old_checkpoint_requires_new_directory(tmp_path):
    (tmp_path / "gepa_state.bin").write_bytes(b"existing checkpoint")
    with pytest.raises(ValueError, match="no manifest"):
        check_checkpoint(tmp_path, {"validation": ["one"]})
    assert not (tmp_path / MANIFEST_FILE).exists()

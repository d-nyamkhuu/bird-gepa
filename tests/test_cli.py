import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import dspy
import httpx
import pytest

from bird_gepa.cli import main, parse_args
from bird_gepa.codex import CodexLM
from bird_gepa.database import Database
from bird_gepa.optimization import EvaluationSummary, OptimizationResult, SearchSettings
from bird_gepa.prepare import prepare_examples
from bird_gepa.program import SEED_SOURCE


@pytest.fixture
def prepared(tmp_path, demo, monkeypatch):
    rows = demo * 2
    (tmp_path / "train.json").write_text(json.dumps(rows[:12]))
    validation = [dict(row, db_id="validation") for row in rows[12:]]
    (tmp_path / "validation.json").write_text(json.dumps(validation))
    (tmp_path / "validation").mkdir()
    shutil.copyfile(tmp_path / "demo/demo.sqlite", tmp_path / "validation/validation.sqlite")
    prepare_examples(tmp_path, tmp_path)

    def unexpected_query(*args, **kwargs):
        pytest.fail("Prepared example loading must not execute SQL")

    monkeypatch.setattr(Database, "query", unexpected_query)
    return tmp_path


def optimization_result():
    summary = EvaluationSummary(total=4, correct=2, score=0.5, queries=4, query_seconds=0.1)
    return OptimizationResult(SimpleNamespace(module_src=SEED_SOURCE), summary, summary)


def test_module_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "bird_gepa.cli", "--help"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert "--prepared" in result.stdout
    assert "--reflection-model" in result.stdout
    assert "--num-threads" in result.stdout
    assert "--checkpoint-dir" in result.stdout
    assert "--skip-baseline" in result.stdout
    assert "--max-predictor-calls" in result.stdout
    assert "--query-penalty" in result.stdout


def test_defaults_use_smaller_validation_and_larger_reflection_batches():
    args = parse_args([])
    assert args.prepared == Path("data/bird/prepared")
    assert args.databases == Path("data/bird/databases")
    assert args.metric_calls == 5500
    assert args.num_threads == 8
    assert args.reflection_minibatch_size == 8
    assert args.validation_size == 300
    assert args.max_predictor_calls == 10
    assert args.query_penalty == 0.01
    assert args.skip_baseline is False
    assert args.checkpoint_dir == args.output.parent / "checkpoint"
    assert args.task_model == "codex/gpt-6-luna"
    assert args.reflection_model == "codex/gpt-6-astra"


def test_explicit_arguments_override_defaults():
    args = parse_args(
        [
            "--prepared",
            "prepared",
            "--metric-calls",
            "20000",
            "--num-threads",
            "2",
            "--reflection-minibatch-size",
            "3",
            "--validation-size",
            "150",
            "--max-predictor-calls",
            "5",
            "--query-penalty",
            "0.02",
            "--skip-baseline",
            "--checkpoint-dir",
            "saved-run",
        ]
    )
    assert args.prepared == Path("prepared")
    assert args.metric_calls == 20000
    assert args.num_threads == 2
    assert args.reflection_minibatch_size == 3
    assert args.validation_size == 150
    assert args.max_predictor_calls == 5
    assert args.query_penalty == 0.02
    assert args.skip_baseline is True
    assert args.checkpoint_dir == Path("saved-run")


@pytest.mark.parametrize("count", ["0", "-1", "1.5", "invalid"])
@pytest.mark.parametrize(
    "option",
    [
        "--metric-calls",
        "--num-threads",
        "--reflection-minibatch-size",
        "--validation-size",
        "--max-predictor-calls",
    ],
)
def test_counts_must_be_positive(option, count):
    with pytest.raises(SystemExit) as exc:
        parse_args([option, count])
    assert exc.value.code == 2


def test_raw_data_option_is_rejected():
    with pytest.raises(SystemExit) as exc:
        parse_args(["--data", "questions.json"])
    assert exc.value.code == 2


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("skip_baseline", [False, True])
def test_main_passes_data_and_models_to_optimizer(
    tmp_path, prepared, monkeypatch, fail, skip_baseline
):
    output = tmp_path / "run" / "solver.py"
    monkeypatch.setattr(
        "sys.argv",
        [
            "bird_gepa.cli",
            "--prepared",
            str(prepared),
            "--databases",
            str(tmp_path),
            "--output",
            str(output),
            *(["--skip-baseline"] if skip_baseline else []),
        ],
    )

    calls = []

    def fake_optimize(train, validation, databases, *, task_lm, reflection_lm, **kwargs):
        calls.append(True)
        assert (len(train), len(validation)) == (12, 4)
        assert task_lm.model == "codex/gpt-6-luna"
        assert reflection_lm.model == "codex/gpt-6-astra"
        for lm in (task_lm, reflection_lm):
            assert isinstance(lm, CodexLM)
        assert kwargs["num_threads"] == 8
        assert kwargs["max_metric_calls"] == 5500
        assert kwargs["settings"] == SearchSettings()
        assert kwargs["skip_baseline"] is skip_baseline
        assert task_lm.cache is True
        assert task_lm.kwargs["timeout"] == 90
        assert reflection_lm.cache is False
        assert reflection_lm.kwargs["timeout"] == 180
        assert kwargs["checkpoint_dir"] == output.parent / "checkpoint"
        if fail:
            raise RuntimeError("inference failed")
        return optimization_result()

    monkeypatch.setattr("bird_gepa.cli.optimize", fake_optimize)
    if fail:
        with pytest.raises(RuntimeError, match="inference failed"):
            main()
    else:
        main()
    assert calls == [True]
    assert output.exists() is not fail
    if not fail:
        assert output.read_text() == SEED_SOURCE
    assert not output.with_suffix(".json").exists()


def test_metric_budget_is_checked_against_actual_validation_size(tmp_path, prepared, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "bird_gepa.cli",
            "--prepared",
            str(prepared),
            "--databases",
            str(tmp_path),
            "--metric-calls",
            "4",
        ],
    )
    with pytest.raises(ValueError, match="metric budget larger than the validation split"):
        main()


def test_metric_budget_uses_subset_size(tmp_path, prepared, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "bird_gepa.cli",
            "--prepared",
            str(prepared),
            "--databases",
            str(tmp_path),
            "--metric-calls",
            "4",
            "--validation-size",
            "2",
            "--output",
            str(tmp_path / "run/solver.py"),
        ],
    )
    calls = []

    def fake_optimize(train, validation, databases, **kwargs):
        calls.append(True)
        assert len(train) == 12
        assert len(validation) == 4
        assert kwargs["settings"].validation_size == 2
        assert kwargs["max_metric_calls"] == 4
        return optimization_result()

    monkeypatch.setattr("bird_gepa.cli.optimize", fake_optimize)
    main()
    assert calls == [True]


@pytest.mark.parametrize("penalty", ["-0.01", "1.01", "nan", "inf"])
def test_query_penalty_must_be_finite_and_bounded(penalty):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--query-penalty", penalty])
    assert exc.value.code == 2


def test_zero_query_penalty_disables_cost_scoring():
    assert parse_args(["--query-penalty", "0"]).query_penalty == 0


def test_main_does_not_require_openai_api_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["bird_gepa.cli"])

    def read_splits(path):
        raise RuntimeError("data loading reached")

    monkeypatch.setattr("bird_gepa.data.read_splits", read_splits)
    with pytest.raises(RuntimeError, match="data loading reached"):
        main()


def test_models_send_responses_requests_without_storage(
    tmp_path, prepared, monkeypatch, codex_login, codex_transport
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "bird_gepa.cli",
            "--prepared",
            str(prepared),
            "--databases",
            str(tmp_path),
            "--output",
            str(tmp_path / "run/solver.py"),
        ],
    )
    requests = []

    auth_file = codex_login()
    token = json.loads(auth_file.read_text())["tokens"]["access_token"]
    monkeypatch.setenv("CODEX_HOME", str(auth_file.parent))

    def handler(request):
        requests.append(request)
        body = json.loads(request.content)
        assert str(request.url) == "https://chatgpt.com/backend-api/codex/responses"
        assert request.headers["Authorization"] == f"Bearer {token}"
        assert body["store"] is False
        assert body["stream"] is True
        assert "max_output_tokens" not in body
        response = {
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": body["model"],
            "output": [
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps({"answer": "OK"}),
                            "annotations": [],
                        }
                    ],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        event = {"type": "response.completed", "sequence_number": 0, "response": response}
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=f"data: {json.dumps(event)}\n\n",
        )

    def optimize(train, validation, databases, *, task_lm, reflection_lm, **kwargs):
        predict = dspy.Predict("question -> answer")
        for lm in (task_lm, reflection_lm):
            predict.set_lm(lm)
            with dspy.context(adapter=dspy.JSONAdapter()):
                assert predict(question="Hello").answer == "OK"
        return optimization_result()

    codex_transport(handler)
    monkeypatch.setattr("bird_gepa.cli.optimize", optimize)
    main()
    assert len(requests) == 2

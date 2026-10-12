import asyncio
import copy
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dspy
import httpx
import pytest

from bird_gepa.codex import CodexLM, auth_headers


def completion(content, *, model="gpt-6.1-sol"):
    return {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "model": model,
            "output": [{"type": "message", "content": [{"type": "output_text", "text": content}]}],
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        },
    }


def stream(events):
    return ": keepalive\n\n" + "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    )


@pytest.fixture
def auth_file(codex_login):
    return codex_login()


@pytest.fixture
def provider(codex_transport):
    requests = []
    events = [completion("AUTH_OK")]
    status = [200]

    def handler(request):
        requests.append(request)
        return httpx.Response(status[0], text=stream(events))

    codex_transport(handler)
    return requests, events, status


def test_request_shape_usage_and_secret_free_state(auth_file, provider):
    requests, _, _ = provider
    lm = CodexLM("codex/gpt-6.1-sol", auth_file=auth_file, cache=False)
    result = lm(
        messages=[{"role": "system", "content": "Be brief"}, {"role": "user", "content": "Hello"}],
        max_tokens=128,
        temperature=0.5,
    )
    assert result == ["AUTH_OK"]
    body = json.loads(requests[0].content)
    assert body == {
        "model": "gpt-6.1-sol",
        "instructions": "Be brief",
        "input": [{"role": "user", "content": "Hello"}],
        "store": False,
        "stream": True,
    }
    token = json.loads(auth_file.read_text())["tokens"]["access_token"]
    assert requests[0].headers["Authorization"] == f"Bearer {token}"
    assert requests[0].headers["ChatGPT-Account-ID"] == "test-account"
    assert requests[0].headers["originator"] == "bird-gepa"
    assert lm.history[-1]["usage"]["prompt_tokens"] == 10
    assert token not in str(lm.dump_state()) + str(lm.history) + str(vars(lm))


def test_reasoning_effort_is_sent(auth_file, provider):
    requests, _, _ = provider
    CodexLM("codex/gpt-6.1-sol", auth_file=auth_file, cache=False)("hi", reasoning_effort="low")
    assert json.loads(requests[0].content)["reasoning"] == {"effort": "low"}


def test_auth_file_defaults_to_codex_home(codex_login, monkeypatch):
    path = codex_login(
        **{
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "eu-account",
                "chatgpt_data_residency": "eu",
            }
        }
    )
    monkeypatch.setenv("CODEX_HOME", str(path.parent))
    headers = auth_headers()
    assert headers["ChatGPT-Account-ID"] == "eu-account"
    assert headers["x-openai-internal-codex-residency"] == "eu"


@pytest.mark.parametrize("contents", [None, "{}", '{"tokens": {"access_token": "opaque"}}'])
def test_missing_login_fails_before_dispatch(tmp_path, provider, contents):
    path = tmp_path / "auth.json"
    if contents is not None:
        path.write_text(contents)
    with pytest.raises(dspy.LMAuthError, match="codex login"):
        CodexLM("codex/gpt-6.1-sol", auth_file=path)("hello")
    assert not provider[0]


def test_expired_login_fails_before_dispatch(codex_login, provider):
    auth_file = codex_login(exp=time.time() - 1)
    with pytest.raises(dspy.LMAuthError, match="expired"):
        CodexLM("codex/gpt-6.1-sol", auth_file=auth_file)("hello")
    assert not provider[0]


def test_credentials_are_reread_on_each_request(codex_login, provider):
    requests, _, _ = provider
    lm = CodexLM("codex/gpt-6.1-sol", auth_file=codex_login(), cache=False)
    lm("hello")
    codex_login(**{"https://api.openai.com/auth": {"chatgpt_account_id": "other-account"}})
    lm("hello")
    assert [r.headers["ChatGPT-Account-ID"] for r in requests] == ["test-account", "other-account"]


def test_cache_and_copies(auth_file, provider):
    requests, _, _ = provider
    lm = CodexLM("codex/gpt-6.1-sol", auth_file=auth_file)
    assert lm("cache-test-unique") == ["AUTH_OK"]
    copied = copy.deepcopy(lm)
    assert copied("cache-test-unique") == ["AUTH_OK"]
    assert len(requests) == 1
    copied("cache-test-unique", rollout_id=1)
    assert len(requests) == 2


@pytest.mark.skipif(not Path("/proc/self/fd").exists(), reason="Requires Linux FD accounting")
def test_disk_cache_closes_connections_from_short_lived_threads(
    auth_file, provider, tmp_path, monkeypatch
):
    cache = dspy.clients.cache.Cache(
        disk_cache_dir=str(tmp_path / "disk-cache"),
        enable_disk_cache=True,
        enable_memory_cache=False,
    )
    cache.disk_cache.close()
    monkeypatch.setattr(dspy, "cache", cache)
    lm = CodexLM("codex/gpt-6.1-sol", auth_file=auth_file)
    requests, _, _ = provider

    def cache_descriptors():
        return sum("disk-cache" in str(fd.resolve()) for fd in Path("/proc/self/fd").iterdir())

    before = cache_descriptors()
    gc.disable()
    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(lambda i: lm(f"disk-cache-{i}"), range(128)))
        assert results == [["AUTH_OK"]] * 128
        assert cache_descriptors() <= before
        assert lm("disk-cache-0") == ["AUTH_OK"]
        assert len(requests) == 128
    finally:
        gc.enable()
        cache.disk_cache.close()
        gc.collect()


@pytest.mark.parametrize(
    "event",
    [
        {"type": "error", "code": "invalid_prompt", "message": "Bad input"},
        {"type": "response.failed", "response": {"error": {"code": "invalid_prompt"}}},
        {"type": "response.incomplete", "response": {"incomplete_details": {"reason": "x"}}},
    ],
)
def test_stream_failures_are_reported(auth_file, provider, event):
    requests, events, _ = provider
    events[:] = [event]
    lm = CodexLM("codex/gpt-6.1-sol", auth_file=auth_file, num_retries=0)
    with pytest.raises(dspy.LMProviderError) as error:
        lm("failed-stream")
    assert json.dumps(event) in str(error.value)
    assert len(requests) == 1


@pytest.mark.parametrize("status", [401, 403, 400])
def test_auth_and_request_errors_are_not_retried(auth_file, provider, status):
    requests, _, statuses = provider
    statuses[0] = status
    lm = CodexLM("codex/gpt-6.1-sol", auth_file=auth_file, cache=False, num_retries=2)
    with pytest.raises(dspy.LMProviderError) as error:
        lm("hello")
    assert isinstance(error.value, dspy.LMAuthError) is (status != 400)
    assert error.value.status == status
    assert len(requests) == 1


@pytest.mark.parametrize("failure", ["stream", "truncated", "timeout", "http500", "http429"])
@pytest.mark.parametrize("recover", [False, True])
def test_transient_failures_retry_and_cache_only_completion(
    auth_file, codex_transport, monkeypatch, failure, recover
):
    requests = []
    sleeps = []

    def handler(request):
        requests.append(request)
        if recover and len(requests) == 3:
            return httpx.Response(200, text=stream([completion("RECOVERED")]))
        if failure == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        if failure.startswith("http"):
            return httpx.Response(int(failure[4:]), json={"error": {"code": "server_error"}})
        if failure == "truncated":
            return httpx.Response(200, text="data: {}\n\n")
        return httpx.Response(200, text=stream([{"type": "error", "code": "server_error"}]))

    codex_transport(handler)
    monkeypatch.setattr("bird_gepa.codex.time.sleep", sleeps.append)
    monkeypatch.setattr("bird_gepa.codex.random.random", lambda: 0)
    lm = CodexLM("codex/gpt-6.1-sol", auth_file=auth_file, num_retries=2)
    if recover:
        assert lm("retry-test") == lm("retry-test") == ["RECOVERED"]
    else:
        with pytest.raises(dspy.LMProviderError):
            lm("retry-test")
    assert len(requests) == 3
    assert sleeps == [1, 2]


def test_async_transport(auth_file, provider):
    lm = CodexLM("codex/gpt-6.1-sol", auth_file=auth_file, cache=False)
    assert asyncio.run(lm.acall("hello")) == ["AUTH_OK"]
    assert asyncio.run(lm.acall("hello again")) == ["AUTH_OK"]
    assert len(provider[0]) == 2


@pytest.mark.parametrize("kwargs", [{"n": 2}, {"tools": []}, {"response_format": {}}])
def test_unsupported_options_fail_before_dispatch(auth_file, provider, kwargs):
    lm = CodexLM("codex/gpt-6.1-sol", auth_file=auth_file, cache=False)
    with pytest.raises(ValueError, match="does not support"):
        lm("hello", **kwargs)
    assert not provider[0]


def test_stream_requires_completion(auth_file, provider):
    provider[1][:] = [{"type": "response.output_text.delta", "delta": "partial"}]
    lm = CodexLM("codex/gpt-6.1-sol", auth_file=auth_file, cache=False, num_retries=0)
    with pytest.raises(dspy.LMProviderError, match="before completion"):
        lm("hello")


def test_output_items_fill_an_empty_completion(auth_file, provider):
    final = completion("AUTH_OK")
    item = final["response"]["output"].pop()
    provider[1][:] = [
        {"type": "response.output_text.delta", "delta": "AUTH_OK"},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        final,
    ]
    lm = CodexLM("codex/gpt-6.1-sol", auth_file=auth_file, cache=False)
    assert lm("hello") == ["AUTH_OK"]
    assert lm.history[-1]["usage"]["completion_tokens"] == 20


def test_dspy_predictor_save_and_reload(auth_file, provider, tmp_path):
    provider[1][:] = [completion(json.dumps({"answer": "AUTH_OK"}))]
    lm = CodexLM("codex/gpt-6.1-sol", auth_file=auth_file, cache=False)
    program = dspy.Predict("question -> answer")
    program.set_lm(lm)
    with dspy.context(adapter=dspy.JSONAdapter()):
        assert program(question="Hello").answer == "AUTH_OK"
    path = tmp_path / "program.json"
    # Runtime providers are reconstructed explicitly.
    program.set_lm(None)
    program.save(path)
    assert json.loads(auth_file.read_text())["tokens"]["access_token"] not in path.read_text()
    assert json.loads(path.read_text())["lm"] is None
    restored = dspy.Predict("question -> answer")
    restored.load(path)
    restored.set_lm(lm)
    with dspy.context(adapter=dspy.JSONAdapter()):
        assert restored(question="Hello again").answer == "AUTH_OK"
    assert len(provider[0]) == 2

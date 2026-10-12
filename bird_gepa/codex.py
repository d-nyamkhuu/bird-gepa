"""A DSPy language model that sends requests with your Codex (ChatGPT) login."""

import asyncio
import base64
import json
import logging
import math
import os
import random
import time
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import dspy
import httpx
from litellm import ModelResponse

logger = logging.getLogger(__name__)

RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

# One connection pool for every LM, copy and thread. httpx clients are thread-safe.
_client = httpx.Client()


def token_claims(token: str) -> dict:
    """Decode a JWT payload without verifying it. The server verifies the token."""
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


def auth_headers(path: Path | None = None) -> dict[str, str]:
    """Build request headers from the login saved by `codex login`.

    As in Hermes Agent, `auth.json` is only read. Refresh tokens are single-use,
    so refreshing here would sign out the Codex CLI. Run `codex login` instead.
    """
    if path is None:
        path = Path(os.environ.get("CODEX_HOME", "~/.codex")) / "auth.json"
    try:
        token = json.loads(path.expanduser().read_text())["tokens"]["access_token"]
        claims = token_claims(token)
        auth = claims["https://api.openai.com/auth"]
        account_id = auth["chatgpt_account_id"]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        raise dspy.LMAuthError("No Codex login found. Run `codex login`.") from None
    if claims.get("exp", math.inf) <= time.time():
        raise dspy.LMAuthError("Codex login expired. Run `codex login`.")

    headers = {
        "Authorization": f"Bearer {token}",
        "ChatGPT-Account-ID": account_id,
        # Third-party clients identify themselves rather than posing as the Codex CLI.
        "originator": "bird-gepa",
        "User-Agent": "bird-gepa",
        "Accept": "text/event-stream",
    }
    # Workspaces with data residency answer 401 unless the request names their region.
    residency = auth.get("chatgpt_data_residency") or auth.get("chatgpt_compute_residency")
    if residency:
        headers["x-openai-internal-codex-residency"] = residency
    return headers


def request_body(
    model: str, prompt: str | None, messages: list | None, reasoning_effort: str | None
) -> dict:
    """Translate chat messages into a Responses API request.

    Codex rejects system messages in `input`, so they become `instructions`.
    It also rejects `max_output_tokens` and `temperature`, so neither is sent.
    """
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    instructions = [m["content"] for m in messages if m["role"] in ("system", "developer")]
    inputs = [m for m in messages if m["role"] not in ("system", "developer")]
    body = {
        "model": model.removeprefix("codex/"),
        "instructions": "\n\n".join(instructions),
        "input": [{"role": m["role"], "content": m["content"]} for m in inputs],
        "store": False,
        "stream": True,
    }
    if reasoning_effort is not None:
        body["reasoning"] = {"effort": reasoning_effort}
    return body


def iter_events(lines: Iterator[str]) -> Iterator[dict]:
    """Yield the JSON payload of each server-sent event."""
    for line in lines:
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            yield json.loads(line[5:])


def read_stream(response: httpx.Response) -> dict:
    """Return the final response object from a Responses API event stream."""
    items: dict[int, dict] = {}
    for event in iter_events(response.iter_lines()):
        kind = event.get("type")
        if kind == "response.output_item.done":
            items[event["output_index"]] = event["item"]
        elif kind == "response.completed":
            # Codex may send the output items only as separate events.
            completed = event["response"]
            completed["output"] = completed.get("output") or [items[i] for i in sorted(items)]
            return completed
        elif kind in ("error", "response.failed", "response.incomplete"):
            raise dspy.LMProviderError(f"Codex request failed: {json.dumps(event)[:1500]}")
    raise dspy.LMProviderError("Codex stream ended before completion.")


def to_model_response(completed: dict) -> ModelResponse:
    """Convert a completed Responses API object into the shape DSPy expects."""
    text = "".join(
        part["text"]
        for item in completed["output"]
        if item.get("type") == "message"
        for part in item["content"]
        if part.get("type") == "output_text"
    )
    usage = completed.get("usage") or {}
    return ModelResponse(
        model=completed.get("model"),
        choices=[{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        usage={
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    )


def is_transient(error: dspy.LMProviderError) -> bool:
    """Retry rate limits, server errors, and dropped streams, but not bad requests."""
    if isinstance(error, dspy.LMAuthError):
        return False
    return error.status is None or error.status in RETRY_STATUS_CODES


class CodexLM(dspy.BaseLM):
    """A text-only DSPy LM using the ChatGPT login saved by `codex login`.

    Credentials are read from `auth.json` on each request, so a login refreshed
    by the Codex CLI is picked up and no token is stored on the LM.
    DSPy's `temperature` and `max_tokens` are accepted but not sent.
    """

    forward_contract = "legacy"

    def __init__(
        self,
        model: str,
        *,
        auth_file: Path | None = None,
        timeout: float = 120,
        cache: bool = True,
        num_retries: int = 3,
    ) -> None:
        super().__init__(model, cache=cache, num_retries=num_retries, timeout=timeout)
        self.auth_file = auth_file

    def forward(
        self, prompt: str | None = None, messages: list | None = None, **kwargs: Any
    ) -> ModelResponse:
        options = {**self.kwargs, **kwargs}
        cache = options.pop("cache", self.cache)
        rollout_id = options.pop("rollout_id", None)
        timeout = options.pop("timeout")
        reasoning_effort = options.pop("reasoning_effort", None)
        options.pop("temperature", None)
        options.pop("max_tokens", None)
        if options:
            raise ValueError(f"CodexLM does not support {', '.join(sorted(options))}.")

        body = request_body(self.model, prompt, messages, reasoning_effort)
        if not cache:
            return self._send_with_retries(body, timeout)

        key = {"provider": "codex", "body": body, "rollout_id": rollout_id}
        # Flex calls models from short-lived threads. Close each thread's disk-cache
        # connection here, instead of leaving it open until garbage collection.
        with dspy.cache.disk_cache if dspy.cache.enable_disk_cache else nullcontext():
            response = dspy.cache.get(key)
            if response is None:
                response = self._send_with_retries(body, timeout)
                dspy.cache.put(key, response)
        return response

    async def aforward(
        self, prompt: str | None = None, messages: list | None = None, **kwargs: Any
    ) -> ModelResponse:
        return await asyncio.to_thread(self.forward, prompt=prompt, messages=messages, **kwargs)

    def _send_with_retries(self, body: dict, timeout: float) -> ModelResponse:
        for attempt in range(self.num_retries):
            try:
                return self._send_single_request(body, timeout)
            except dspy.LMProviderError as error:
                if not is_transient(error):
                    raise
                logger.warning(
                    "Retrying Codex request (%d/%d): %s", attempt + 1, self.num_retries, error
                )
                time.sleep(min(2**attempt, 16) + random.random())
        return self._send_single_request(body, timeout)

    def _send_single_request(self, body: dict, timeout: float) -> ModelResponse:
        headers = auth_headers(self.auth_file)
        try:
            with _client.stream(
                "POST", RESPONSES_URL, json=body, headers=headers, timeout=timeout
            ) as response:
                if response.status_code in (401, 403):
                    raise dspy.LMAuthError(
                        "Codex rejected this login. Run `codex login`.", status=response.status_code
                    )
                if response.status_code != 200:
                    response.read()
                    status = response.status_code
                    raise dspy.LMProviderError(
                        f"Codex request failed (HTTP {status}): {response.text[:1500]}",
                        status=status,
                    )
                return to_model_response(read_stream(response))
        except httpx.HTTPError as error:
            raise dspy.LMProviderError(f"Codex request failed: {error!r}") from error

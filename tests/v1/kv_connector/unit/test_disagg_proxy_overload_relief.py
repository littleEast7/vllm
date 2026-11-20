# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for examples/.../disagg_proxy_overload_relief.py against fake P and D."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.cpu_test

PROXY_PATH = (
    Path(__file__).parents[4]
    / "examples/disaggregated/disaggregated_serving/disagg_proxy_overload_relief.py"
)
PROMPT_TOKEN_IDS = [11, 12, 13]
P_KV_PARAMS = {"do_remote_prefill": True, "remote_engine_id": "p"}


@pytest.fixture
def proxy():
    spec = importlib.util.spec_from_file_location("overload_relief_proxy", PROXY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve annotations through sys.modules.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def _chunk(content=None, **choice):
    delta = {"content": content} if content is not None else {}
    return {"choices": [{"index": 0, "delta": delta, **choice}]}


def _sse(chunks):
    body = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)
    return httpx.Response(
        200, content=body + b"data: [DONE]\n\n", headers={"content-type": "text/sse"}
    )


class FakeBackends:
    """P answers every prefill; D declines offers unless `d_has_prefix`."""

    def __init__(self, proxy, d_has_prefix, d_offer_status=200):
        self.proxy = proxy
        self.d_has_prefix = d_has_prefix
        self.d_offer_status = d_offer_status
        self.p_requests: list[dict] = []
        self.d_requests: list[dict] = []

    def prefiller(self, request: httpx.Request) -> httpx.Response:
        self.p_requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "kv_transfer_params": P_KV_PARAMS,
                "prompt_token_ids": PROMPT_TOKEN_IDS,
            },
        )

    def decoder(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.d_requests.append(body)
        if not body.get("stream"):
            return httpx.Response(200, json={"choices": [{"message": "ok"}]})
        first = {"prompt_token_ids": PROMPT_TOKEN_IDS, **_chunk("", role="assistant")}
        offered = (
            self.proxy.MAX_UNCACHED_PROMPT_TOKENS_KEY in body["kv_transfer_params"]
        )
        if offered and self.d_offer_status != 200:
            return httpx.Response(self.d_offer_status, json={"error": "busy"})
        if offered and not self.d_has_prefix:
            declined = _chunk(
                finish_reason="stop",
                stop_reason=self.proxy.UNCACHED_PROMPT_LIMIT_STOP_REASON,
            )
            return _sse([first, declined])
        return _sse(
            [
                first,
                _chunk("Hel", token_ids=[1]),
                _chunk("lo", token_ids=[2]),
                _chunk(finish_reason="stop"),
            ]
        )


def _make_state(proxy, backends, gate_threshold):
    args = argparse.Namespace(
        gate_threshold=gate_threshold,
        uncached_limit=160,
        prefiller_host="p",
        prefiller_port=1,
        decoder_host="d",
        decoder_port=2,
    )
    state = proxy.ProxyState(args)
    state.prefiller = httpx.AsyncClient(
        base_url="http://p/v1", transport=httpx.MockTransport(backends.prefiller)
    )
    state.decoder = httpx.AsyncClient(
        base_url="http://d/v1", transport=httpx.MockTransport(backends.decoder)
    )
    return state


async def _run(proxy, state, body):
    pieces = [p async for p in proxy.handle_stream(state, body, "req-0")]
    events = b"".join(pieces).split(b"\n\n")
    chunks = [json.loads(e[6:]) for e in events if e and e != b"data: [DONE]"]
    text = "".join(
        c["choices"][0]["delta"].get("content") or "" for c in chunks if "choices" in c
    )
    return chunks, text, pieces


BODY = {
    "model": "m",
    "messages": [{"role": "user", "content": "hi"}],
    "stream": True,
    "max_tokens": 8,
    "min_tokens": 2,
}


@pytest.mark.asyncio
async def test_gate_closed_takes_prefill_then_decode(proxy):
    backends = FakeBackends(proxy, d_has_prefix=True)
    state = _make_state(proxy, backends, gate_threshold=1)
    chunks, text, pieces = await _run(proxy, state, BODY)

    assert text == "Hello"
    assert pieces[-1] == b"data: [DONE]\n\n"
    (p_body,) = backends.p_requests
    assert p_body["max_tokens"] == 1 and p_body["stream"] is False
    assert p_body["kv_transfer_params"]["do_remote_decode"] is True
    assert "min_tokens" not in p_body
    (d_body,) = backends.d_requests
    assert d_body["kv_transfer_params"] == {
        **P_KV_PARAMS,
        "prompt_token_ids": PROMPT_TOKEN_IDS,
    }
    assert d_body["max_tokens"] == 8 and d_body["min_tokens"] == 2
    # The client did not ask for token ids.
    assert all("prompt_token_ids" not in c for c in chunks)
    assert all("token_ids" not in c["choices"][0] for c in chunks)
    assert state.prefills_outstanding == 0
    assert (state.num_prefilled, state.num_served_by_decoder) == (1, 0)


@pytest.mark.asyncio
async def test_gate_open_and_decoder_has_prefix_skips_prefill(proxy):
    backends = FakeBackends(proxy, d_has_prefix=True)
    state = _make_state(proxy, backends, gate_threshold=0)
    _, text, _ = await _run(proxy, state, BODY)

    assert text == "Hello"
    assert not backends.p_requests
    (d_body,) = backends.d_requests
    assert d_body["kv_transfer_params"] == {proxy.MAX_UNCACHED_PROMPT_TOKENS_KEY: 160}
    assert (state.num_served_by_decoder, state.num_declined) == (1, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("d_offer_status", [200, 503])
async def test_gate_open_and_decoder_declines_falls_back(proxy, d_offer_status):
    backends = FakeBackends(proxy, d_has_prefix=False, d_offer_status=d_offer_status)
    state = _make_state(proxy, backends, gate_threshold=0)
    chunks, text, pieces = await _run(proxy, state, BODY)

    # The client sees one clean stream: nothing from the declined attempt.
    assert text == "Hello"
    assert pieces.count(b"data: [DONE]\n\n") == 1
    assert sum("role" in c["choices"][0] for c in chunks) == 1
    assert all(
        c["choices"][0].get("stop_reason") != proxy.UNCACHED_PROMPT_LIMIT_STOP_REASON
        for c in chunks
    )
    assert len(backends.p_requests) == 1
    offer, decode = backends.d_requests
    assert proxy.MAX_UNCACHED_PROMPT_TOKENS_KEY in offer["kv_transfer_params"]
    assert decode["kv_transfer_params"] == {
        **P_KV_PARAMS,
        "prompt_token_ids": PROMPT_TOKEN_IDS,
    }
    assert (state.num_declined, state.num_prefilled) == (1, 1)


@pytest.mark.asyncio
async def test_token_ids_kept_when_client_asks(proxy):
    backends = FakeBackends(proxy, d_has_prefix=True)
    state = _make_state(proxy, backends, gate_threshold=0)
    chunks, _, _ = await _run(proxy, state, {**BODY, "return_token_ids": True})
    assert chunks[0]["prompt_token_ids"] == PROMPT_TOKEN_IDS
    assert [c["choices"][0].get("token_ids") for c in chunks[1:3]] == [[1], [2]]


def test_gate_tracks_outstanding_prefills(proxy):
    state = _make_state(proxy, FakeBackends(proxy, d_has_prefix=True), 2)
    assert not state.gate_open()
    state.prefills_outstanding = 2
    assert state.gate_open()

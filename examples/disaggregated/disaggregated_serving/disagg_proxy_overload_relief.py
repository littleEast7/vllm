# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disaggregated Prefill/Decode proxy with prefill overload relief.

The usual P/D flow sends every request to the Prefill instance (P) first, so a
few long prompts queueing on P delay the first token of every other request,
including ones whose KV the Decode instance (D) could obtain without P: from
its own prefix cache, or from a shared cache tier it reads through a KV
connector.

This proxy keeps the usual flow while P keeps up. While P is backed up, it
offers each request to D first, with a limit on how many prompt tokens D may
have to compute itself:

    Client ──► Proxy ──► P ──► D            P keeps up, or D declined
    Client ──► Proxy ──► D                  P backed up, and D has the prefix

Per-request flow (streaming `/v1/chat/completions`):
    1. If fewer than `--gate-threshold` prefills of this proxy are outstanding
       on P, go to step 4.
    2. Send the request to D with
       `kv_transfer_params = {"max_uncached_prompt_tokens": --uncached-limit}`.
    3. D serves the request if at most that many prompt tokens are missing from
       its prefix cache and KV connector. Otherwise it declines: the stream
       ends with `stop_reason == "uncached_prompt_limit"`, no tokens, and the
       prompt token ids. Nothing has been sent to the client yet.
    4. Send the request to P (`max_tokens=1`, `do_remote_decode`), then to D
       with P's `kv_transfer_params`. The prompt is tokenized once: the ids
       come from D's decline or from P, see "Reusing prefill token ids on
       decode" in docs/features/disagg_prefill.md.

Decode's inter-token latency grows with the prefill work it takes on, so keep
`--uncached-limit` small (a few hundred tokens at most).

Usage:
    python disagg_proxy_overload_relief.py \\
        --host 0.0.0.0 --port 8000 \\
        --prefiller-host 10.0.0.1 --prefiller-port 8100 \\
        --decoder-host 10.0.0.2 --decoder-port 8200 \\
        --gate-threshold 4 --uncached-limit 160
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("disagg_proxy_overload_relief")

MAX_UNCACHED_PROMPT_TOKENS_KEY = "max_uncached_prompt_tokens"
UNCACHED_PROMPT_LIMIT_STOP_REASON = "uncached_prompt_limit"

PREFILL_KV_TRANSFER_PARAMS = {
    "do_remote_decode": True,
    "do_remote_prefill": False,
    "remote_engine_id": None,
    "remote_block_ids": None,
    "remote_host": None,
    "remote_port": None,
}
# Not supported with max_tokens=1 on P; D still gets them.
DECODE_ONLY_FIELDS = ("min_tokens", "min_completion_tokens", "stream_options")


@dataclass
class RelayResult:
    declined: bool = False
    prompt_token_ids: list[int] | None = None


class ProxyState:
    def __init__(self, args: argparse.Namespace):
        self.gate_threshold: int = args.gate_threshold
        self.uncached_limit: int = args.uncached_limit
        timeout = httpx.Timeout(None, connect=10.0)
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        self.prefiller = httpx.AsyncClient(
            base_url=f"http://{args.prefiller_host}:{args.prefiller_port}/v1",
            timeout=timeout,
            limits=limits,
        )
        self.decoder = httpx.AsyncClient(
            base_url=f"http://{args.decoder_host}:{args.decoder_port}/v1",
            timeout=timeout,
            limits=limits,
        )
        # Prefills this proxy has sent to P that have not returned yet.
        self.prefills_outstanding = 0
        self.num_served_by_decoder = 0
        self.num_declined = 0
        self.num_prefilled = 0

    def gate_open(self) -> bool:
        return self.prefills_outstanding >= self.gate_threshold


def _headers(request_id: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }


def _sse(obj: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"


async def prefill(
    state: ProxyState, body: dict[str, Any], request_id: str
) -> dict[str, Any]:
    """Run the prompt on P and return its response."""
    p_body = {k: v for k, v in body.items() if k not in DECODE_ONLY_FIELDS}
    p_body.update(stream=False, max_tokens=1, return_token_ids=True)
    if "max_completion_tokens" in p_body:
        p_body["max_completion_tokens"] = 1
    p_body["kv_transfer_params"] = dict(PREFILL_KV_TRANSFER_PARAMS)
    state.prefills_outstanding += 1
    try:
        response = await state.prefiller.post(
            "/chat/completions", json=p_body, headers=_headers(request_id)
        )
        response.raise_for_status()
        return response.json()
    finally:
        state.prefills_outstanding -= 1


async def relay_decoder_stream(
    state: ProxyState,
    d_body: dict[str, Any],
    request_id: str,
    result: RelayResult,
    *,
    can_decline: bool,
    keep_token_ids: bool,
) -> AsyncIterator[bytes]:
    """Relay D's SSE stream to the client.

    With `can_decline`, nothing is yielded before D's first token, so that a
    decline (reported in `result`) leaves the client's stream untouched.
    """
    held: list[bytes] = []
    started = not can_decline
    buffer = b""
    async with state.decoder.stream(
        "POST", "/chat/completions", json=d_body, headers=_headers(request_id)
    ) as response:
        if response.status_code != 200:
            detail = (await response.aread()).decode(errors="replace")[:500]
            if can_decline:
                logger.warning("Decoder refused the offer: %s", detail)
                result.declined = True
                return
            yield _sse({"error": {"message": detail, "code": response.status_code}})
            return
        async for data in response.aiter_bytes():
            buffer += data
            while b"\n\n" in buffer:
                event, buffer = buffer.split(b"\n\n", 1)
                piece = event + b"\n\n"
                has_token = False
                if event.startswith(b"data: ") and event != b"data: [DONE]":
                    chunk = json.loads(event[6:])
                    if "error" in chunk and not started:
                        logger.warning("Decoder failed the offer: %s", chunk["error"])
                        result.declined = True
                        return
                    if chunk.get("prompt_token_ids") is not None:
                        result.prompt_token_ids = chunk["prompt_token_ids"]
                    if not keep_token_ids:
                        chunk.pop("prompt_token_ids", None)
                    for choice in chunk.get("choices") or []:
                        if (
                            can_decline
                            and choice.get("stop_reason")
                            == UNCACHED_PROMPT_LIMIT_STOP_REASON
                        ):
                            result.declined = True
                            return
                        delta = choice.get("delta") or {}
                        has_token = has_token or bool(
                            choice.get("token_ids")
                            or delta.get("content")
                            or delta.get("reasoning")
                            or delta.get("tool_calls")
                        )
                        if not keep_token_ids:
                            choice.pop("token_ids", None)
                    piece = _sse(chunk)
                if has_token and not started:
                    started = True
                    for held_piece in held:
                        yield held_piece
                    held = []
                if started:
                    yield piece
                else:
                    held.append(piece)
    # The stream ended without a token (e.g. an immediate EOS) and no decline.
    for held_piece in held:
        yield held_piece


async def handle_stream(
    state: ProxyState, body: dict[str, Any], request_id: str
) -> AsyncIterator[bytes]:
    keep_token_ids = bool(body.get("return_token_ids"))
    prompt_token_ids = None

    if state.gate_open():
        d_body = dict(body)
        d_body["return_token_ids"] = True
        d_body["kv_transfer_params"] = {
            MAX_UNCACHED_PROMPT_TOKENS_KEY: state.uncached_limit
        }
        result = RelayResult()
        async for piece in relay_decoder_stream(
            state,
            d_body,
            request_id,
            result,
            can_decline=True,
            keep_token_ids=keep_token_ids,
        ):
            yield piece
        if not result.declined:
            state.num_served_by_decoder += 1
            return
        state.num_declined += 1
        prompt_token_ids = result.prompt_token_ids

    try:
        p_response = await prefill(state, body, request_id)
    except httpx.HTTPError as e:
        logger.exception("Prefill failed")
        yield _sse({"error": {"message": f"prefill failed: {e!r}", "code": 502}})
        return
    state.num_prefilled += 1
    kv_transfer_params = dict(p_response.get("kv_transfer_params") or {})
    prompt_token_ids = prompt_token_ids or p_response.get("prompt_token_ids")
    if prompt_token_ids:
        # D skips chat templating and tokenization.
        kv_transfer_params["prompt_token_ids"] = prompt_token_ids
    d_body = dict(body)
    d_body["kv_transfer_params"] = kv_transfer_params
    async for piece in relay_decoder_stream(
        state,
        d_body,
        request_id,
        RelayResult(),
        can_decline=False,
        keep_token_ids=keep_token_ids,
    ):
        yield piece


async def handle_non_stream(
    state: ProxyState, body: dict[str, Any], request_id: str
) -> JSONResponse:
    """Non-streaming requests always take the P -> D path."""
    p_response = await prefill(state, body, request_id)
    state.num_prefilled += 1
    d_body = dict(body)
    d_body["kv_transfer_params"] = p_response.get("kv_transfer_params") or {}
    response = await state.decoder.post(
        "/chat/completions", json=d_body, headers=_headers(request_id)
    )
    return JSONResponse(response.json(), status_code=response.status_code)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.proxy = ProxyState(global_args)
    yield
    await app.state.proxy.prefiller.aclose()
    await app.state.proxy.decoder.aclose()


app = FastAPI(lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    state: ProxyState = request.app.state.proxy
    body = await request.json()
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    if not body.get("stream"):
        try:
            return await handle_non_stream(state, body, request_id)
        except httpx.HTTPError as e:
            return JSONResponse(
                {"error": {"message": f"upstream failed: {e!r}", "code": 502}},
                status_code=502,
            )
    return StreamingResponse(
        handle_stream(state, body, request_id), media_type="text/event-stream"
    )


@app.get("/healthcheck")
async def healthcheck(request: Request):
    state: ProxyState = request.app.state.proxy
    return {
        "status": "ok",
        "prefills_outstanding": state.prefills_outstanding,
        "gate_open": state.gate_open(),
        "served_by_decoder": state.num_served_by_decoder,
        "declined_by_decoder": state.num_declined,
        "prefilled": state.num_prefilled,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefiller-host", type=str, default="localhost")
    parser.add_argument("--prefiller-port", type=int, default=8100)
    parser.add_argument("--decoder-host", type=str, default="localhost")
    parser.add_argument("--decoder-port", type=int, default=8200)
    parser.add_argument(
        "--gate-threshold",
        type=int,
        default=4,
        help="Offer requests to the decoder first while this many prefills "
        "are outstanding on the prefiller.",
    )
    parser.add_argument(
        "--uncached-limit",
        type=int,
        default=160,
        help="The decoder serves an offered request only if at most this "
        "many prompt tokens are missing from its caches.",
    )
    return parser.parse_args()


global_args: argparse.Namespace

if __name__ == "__main__":
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)

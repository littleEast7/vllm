# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler tests for `kv_transfer_params["max_uncached_prompt_tokens"]`."""

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import (
    MAX_UNCACHED_PROMPT_TOKENS_KEY,
    UNCACHED_PROMPT_LIMIT_STOP_REASON,
)
from vllm.v1.engine import FinishReason
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler, mock_kv

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
NUM_TOKENS = 4 * BLOCK_SIZE
MATCHED_TOKENS = 2 * BLOCK_SIZE


def _request(limit, req_id="0", prompt=0):
    # `create_requests` fills request i's prompt with token id i.
    request = create_requests(
        num_requests=prompt + 1,
        num_tokens=NUM_TOKENS,
        block_size=BLOCK_SIZE,
        req_ids=[f"{req_id}-unused-{i}" for i in range(prompt)] + [req_id],
    )[-1]
    if limit is not None:
        request.kv_transfer_params = {MAX_UNCACHED_PROMPT_TOKENS_KEY: limit}
    return request


def _run_one_step(scheduler, request):
    output = scheduler.schedule()
    assert output.num_scheduled_tokens.keys() == {request.request_id}
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=[request.request_id],
            sampled_token_ids=[[1000]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    return output


def _assert_declined(scheduler, request):
    free_blocks = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    output = scheduler.schedule()
    assert not output.num_scheduled_tokens
    assert not scheduler.running
    # Nothing was allocated for the declined request.
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_blocks

    engine_core_outputs = scheduler.update_from_output(
        output, EMPTY_MODEL_RUNNER_OUTPUT
    )
    (out,) = engine_core_outputs[request.client_index].outputs
    assert out.request_id == request.request_id
    assert out.new_token_ids == []
    assert out.finish_reason == FinishReason.STOP
    assert out.stop_reason == UNCACHED_PROMPT_LIMIT_STOP_REASON
    assert out.error is None

    assert request.status == RequestStatus.FINISHED_STOPPED
    assert not scheduler.uncached_limit_declined_reqs
    assert scheduler.get_num_unfinished_requests() == 0
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_blocks


def test_declined_when_uncached_prompt_exceeds_limit():
    scheduler = create_scheduler(enable_prefix_caching=True, block_size=BLOCK_SIZE)
    request = _request(limit=NUM_TOKENS - 1)
    scheduler.add_request(request)
    _assert_declined(scheduler, request)


@pytest.mark.parametrize("limit", [None, NUM_TOKENS, "64", True])
def test_served_without_limit_or_within_it(limit):
    """No limit, a limit that is met, and non-integer limits (ignored)."""
    scheduler = create_scheduler(enable_prefix_caching=True, block_size=BLOCK_SIZE)
    request = _request(limit=limit)
    scheduler.add_request(request)
    output = _run_one_step(scheduler, request)
    assert output.num_scheduled_tokens[request.request_id] == NUM_TOKENS
    assert request.status == RequestStatus.RUNNING


def test_local_prefix_cache_hit_counts_towards_limit():
    scheduler = create_scheduler(enable_prefix_caching=True, block_size=BLOCK_SIZE)
    first = _request(limit=None, req_id="first")
    scheduler.add_request(first)
    _run_one_step(scheduler, first)

    # The same prompt again: all but the last block is in the local prefix cache.
    cached = _request(limit=BLOCK_SIZE, req_id="cached")
    scheduler.add_request(cached)
    output = scheduler.schedule()
    assert 0 < output.num_scheduled_tokens[cached.request_id] <= BLOCK_SIZE

    # The same limit turns away a prompt that is not cached.
    cold = _request(limit=BLOCK_SIZE, req_id="cold", prompt=1)
    scheduler.add_request(cold)
    output = scheduler.schedule()
    assert cold.request_id not in output.num_scheduled_tokens
    assert cold.request_id in scheduler.uncached_limit_declined_reqs


@pytest.mark.parametrize(
    "limit, declined",
    [(NUM_TOKENS - MATCHED_TOKENS, False), (NUM_TOKENS - MATCHED_TOKENS - 1, True)],
)
def test_kv_connector_hit_counts_towards_limit(limit, declined):
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        use_kv_connector=mock_kv(matched_tokens=MATCHED_TOKENS, is_async=False),
        block_size=BLOCK_SIZE,
    )
    request = _request(limit=limit)
    scheduler.add_request(request)
    if declined:
        _assert_declined(scheduler, request)
    else:
        output = scheduler.schedule()
        assert (
            output.num_scheduled_tokens[request.request_id]
            == NUM_TOKENS - MATCHED_TOKENS
        )


def test_other_requests_are_unaffected_by_a_declined_one():
    scheduler = create_scheduler(enable_prefix_caching=True, block_size=BLOCK_SIZE)
    declined = _request(limit=0, req_id="declined")
    served = _request(limit=None, req_id="served", prompt=1)
    scheduler.add_request(declined)
    scheduler.add_request(served)

    output = scheduler.schedule()
    assert output.num_scheduled_tokens.keys() == {served.request_id}
    engine_core_outputs = scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=[served.request_id],
            sampled_token_ids=[[1000]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    by_id = {
        out.request_id: out
        for client in engine_core_outputs.values()
        for out in client.outputs
    }
    assert by_id[declined.request_id].stop_reason == UNCACHED_PROMPT_LIMIT_STOP_REASON
    assert by_id[served.request_id].new_token_ids == [1000]
    assert served.status == RequestStatus.RUNNING


def test_preempted_request_is_not_declined_on_resume():
    scheduler = create_scheduler(enable_prefix_caching=True, block_size=BLOCK_SIZE)
    request = _request(limit=NUM_TOKENS)
    scheduler.add_request(request)
    _run_one_step(scheduler, request)

    # Resuming recomputes the prompt plus an output token: over the limit, but the
    # request was already admitted.
    scheduler.reset_prefix_cache(reset_running_requests=True)
    assert request.status == RequestStatus.PREEMPTED
    output = scheduler.schedule()
    assert request.request_id in output.num_scheduled_tokens
    assert not scheduler.uncached_limit_declined_reqs

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end tests for jump-forward decoding with structured outputs.

Verifies that when ``enable_jump_decoding=True``, the guidance backend
produces valid structured output consistent with the non-jump baseline,
and that logprobs are correctly synthesised for deterministic tokens.
"""

import gc
import json

import jsonschema
import torch

from tests.utils import single_gpu_only
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

MODEL = "facebook/opt-125m"

PROMPTS = [
    "Generate a person profile in JSON:\n",
    "Output a JSON object with name and age:\n",
]

SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}

# Shared LLM kwargs — keep gpu_memory_utilization low so successive
# LLM instances within the same pytest process can share the GPU.
_LLM_KWARGS = dict(
    model=MODEL,
    max_model_len=512,
    enforce_eager=True,
    gpu_memory_utilization=0.3,
)

# disable_any_whitespace prevents the model from emitting excessive
# newlines / spaces that would exhaust max_tokens before the JSON
# object is complete (opt-125m tends to do this).
_SO_JUMP = {
    "backend": "guidance",
    "enable_jump_decoding": True,
    "disable_any_whitespace": True,
}
_SO_BASELINE = {
    "backend": "guidance",
    "enable_jump_decoding": False,
    "disable_any_whitespace": True,
}


def _make_sampling_params(schema: dict, logprobs: int | None = None):
    return SamplingParams(
        temperature=0.0,
        max_tokens=200,
        logprobs=logprobs,
        structured_outputs=StructuredOutputsParams(json=schema),
    )


def _generate(llm: LLM, prompts: list[str], params: SamplingParams):
    return llm.generate(prompts, params)


def _cleanup(llm: LLM) -> None:
    """Free GPU memory held by an LLM instance.

    The EngineCore runs in a child process; deleting the Python object
    sends the shutdown signal.  gc.collect() + empty_cache() flush
    any remaining host-side references and the CUDA allocator cache.
    """
    del llm
    gc.collect()
    torch.cuda.empty_cache()


@single_gpu_only
def test_jump_forward_produces_valid_json():
    """Output with jump-forward enabled must be valid JSON matching
    the schema."""
    llm = LLM(**_LLM_KWARGS, structured_outputs_config=_SO_JUMP)
    params = _make_sampling_params(SIMPLE_SCHEMA)
    results = _generate(llm, PROMPTS, params)
    _cleanup(llm)

    for result in results:
        text = result.outputs[0].text
        parsed = json.loads(text)
        jsonschema.validate(parsed, SIMPLE_SCHEMA)


@single_gpu_only
def test_jump_forward_matches_baseline():
    """Jump-forward output must be valid JSON just like the baseline.
    Both should conform to the schema, though exact tokens may differ
    due to the different decoding paths."""
    baseline_llm = LLM(**_LLM_KWARGS, structured_outputs_config=_SO_BASELINE)
    params = _make_sampling_params(SIMPLE_SCHEMA)
    baseline_results = _generate(baseline_llm, PROMPTS, params)
    _cleanup(baseline_llm)

    jump_llm = LLM(**_LLM_KWARGS, structured_outputs_config=_SO_JUMP)
    jump_results = _generate(jump_llm, PROMPTS, params)
    _cleanup(jump_llm)

    for baseline_out, jump_out in zip(baseline_results, jump_results):
        baseline_text = baseline_out.outputs[0].text
        jump_text = jump_out.outputs[0].text

        # Both must produce valid JSON conforming to the schema.
        baseline_parsed = json.loads(baseline_text)
        jump_parsed = json.loads(jump_text)
        jsonschema.validate(baseline_parsed, SIMPLE_SCHEMA)
        jsonschema.validate(jump_parsed, SIMPLE_SCHEMA)


@single_gpu_only
def test_jump_forward_logprobs():
    """When logprobs are requested, ff_tokens should have synthetic
    logprobs (0.0 for the deterministic token)."""
    llm = LLM(**_LLM_KWARGS, structured_outputs_config=_SO_JUMP)
    params = _make_sampling_params(SIMPLE_SCHEMA, logprobs=3)
    results = _generate(llm, PROMPTS, params)
    _cleanup(llm)

    for result in results:
        text = result.outputs[0].text
        # Output must still be valid JSON.
        parsed = json.loads(text)
        jsonschema.validate(parsed, SIMPLE_SCHEMA)

        # Every generated token must have logprobs.
        logprobs_list = result.outputs[0].logprobs
        assert logprobs_list is not None
        assert len(logprobs_list) == len(result.outputs[0].token_ids)

        for token_logprobs in logprobs_list:
            # Each position should have at least 1 entry.
            assert len(token_logprobs) >= 1


@single_gpu_only
def test_jump_forward_with_max_tokens():
    """Jump-forward must respect max_tokens — output should not exceed
    the limit even when grammar can produce many deterministic tokens."""
    max_tokens = 15
    llm = LLM(**_LLM_KWARGS, structured_outputs_config=_SO_JUMP)
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        structured_outputs=StructuredOutputsParams(json=SIMPLE_SCHEMA),
    )
    results = _generate(llm, PROMPTS, params)
    _cleanup(llm)

    for result in results:
        assert len(result.outputs[0].token_ids) <= max_tokens

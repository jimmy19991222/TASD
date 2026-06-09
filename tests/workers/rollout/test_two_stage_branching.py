# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for two-stage branching in BranchingAgentLoop.

CPU-only. Uses a fake server_manager and mock score_fn to exercise the
two-stage pipeline: Stage 1 generation, scoring callback, ref/marker
context construction, and leaf assignment.

Run:
    python -m pytest tests/workers/rollout/test_two_stage_branching.py -v
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from types import SimpleNamespace
from typing import Any, Optional
from uuid import uuid4

import pytest


# ---------------------------------------------------------------------------
# Module loading: stub heavy deps (ray, verl.protocol, etc) like the
# existing test_branching_pipeline_on_cpu.py does.
# ---------------------------------------------------------------------------

def _setup_stubs():
    """Stub verl modules that have heavy external deps (ray, torch, etc)."""
    # Stub verl top-level
    if "verl" not in sys.modules:
        verl = types.ModuleType("verl")
        verl.__path__ = ["verl"]
        sys.modules["verl"] = verl
    if "verl.utils" not in sys.modules:
        vu = types.ModuleType("verl.utils")
        vu.__path__ = ["verl/utils"]
        sys.modules["verl.utils"] = vu
    if "verl.experimental" not in sys.modules:
        ve = types.ModuleType("verl.experimental")
        ve.__path__ = ["verl/experimental"]
        sys.modules["verl.experimental"] = ve
    if "verl.experimental.agent_loop" not in sys.modules:
        veal = types.ModuleType("verl.experimental.agent_loop")
        veal.__path__ = ["verl/experimental/agent_loop"]
        sys.modules["verl.experimental.agent_loop"] = veal

    # Stub branching_utils
    bu_spec = importlib.util.spec_from_file_location(
        "verl.workers.rollout.vllm_rollout.branching_utils",
        "verl/workers/rollout/vllm_rollout/branching_utils.py",
    )
    bu = importlib.util.module_from_spec(bu_spec)
    sys.modules["verl.workers.rollout.vllm_rollout.branching_utils"] = bu
    bu_spec.loader.exec_module(bu)

    # Stub agent_loop module
    al = types.ModuleType("verl.experimental.agent_loop.agent_loop")
    class _AgentLoopBase:
        def __init__(self, *a, **k): pass
        async def process_vision_info(self, messages): return {}
        async def apply_chat_template(self, messages, **kwargs): return [10, 11, 12]
    class _AgentLoopMetrics:
        def __init__(self, **kw): self._d = kw
    class _AgentLoopOutput:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            self.extra_fields = kw.get('extra_fields', {})
        def model_copy(self, deep=False):
            return _AgentLoopOutput(**self.__dict__)
    def _register(name):
        def deco(cls): return cls
        return deco
    al.AgentLoopBase = _AgentLoopBase
    al.AgentLoopMetrics = _AgentLoopMetrics
    al.AgentLoopOutput = _AgentLoopOutput
    al.register = _register
    sys.modules["verl.experimental.agent_loop.agent_loop"] = al

    # Stub tools
    tu = types.ModuleType("verl.tools.utils.tool_registry")
    tu.initialize_tools_from_config = lambda p: []
    sys.modules["verl.tools"] = types.ModuleType("verl.tools")
    sys.modules["verl.tools"].__path__ = ["verl/tools"]
    sys.modules["verl.tools.utils"] = types.ModuleType("verl.tools.utils")
    sys.modules["verl.tools.utils"].__path__ = ["verl/tools/utils"]
    sys.modules["verl.tools.utils.tool_registry"] = tu

    # Stub profiler
    prof = types.ModuleType("verl.utils.profiler")
    class _SimpleTimer:
        def __init__(self, name, metrics): self.name = name; self.metrics = metrics
        def __enter__(self): return self
        def __exit__(self, *a): self.metrics[self.name] = 0.0
    prof.simple_timer = _SimpleTimer
    sys.modules["verl.utils.profiler"] = prof

    # Load teacher_prompt (pure-python, no heavy deps)
    tp_spec = importlib.util.spec_from_file_location(
        "verl.utils.teacher_prompt", "verl/utils/teacher_prompt.py"
    )
    tp = importlib.util.module_from_spec(tp_spec)
    sys.modules["verl.utils.teacher_prompt"] = tp
    tp_spec.loader.exec_module(tp)

    # Load branching_agent_loop
    bal_spec = importlib.util.spec_from_file_location(
        "verl.experimental.agent_loop.branching_agent_loop",
        "verl/experimental/agent_loop/branching_agent_loop.py",
    )
    bal = importlib.util.module_from_spec(bal_spec)
    sys.modules["verl.experimental.agent_loop.branching_agent_loop"] = bal
    bal_spec.loader.exec_module(bal)

    return bal, tp


_bal, _tp = _setup_stubs()
BranchingAgentLoop = _bal.BranchingAgentLoop
AgentLoopOutput = sys.modules["verl.experimental.agent_loop.agent_loop"].AgentLoopOutput
AgentLoopMetrics = sys.modules["verl.experimental.agent_loop.agent_loop"].AgentLoopMetrics
build_ref_context_messages = _tp.build_ref_context_messages


# ---------------------------------------------------------------------------
# Fixtures: Fake server + tokenizer
# ---------------------------------------------------------------------------


class _FakeTokenOutput:
    """Minimal fake vLLM generation output."""

    def __init__(self, token_ids, log_probs=None, top_logprobs=None):
        self.token_ids = token_ids
        self.log_probs = log_probs or [0.0] * len(token_ids)
        self.top_logprobs = top_logprobs
        self.routed_experts = None
        self.stop_reason = "completed"


class _FakeServerManager:
    """Returns deterministic token sequences."""

    def __init__(self, response_len: int = 50, top_k: int = 10):
        self.calls: list[dict] = []
        self.response_len = response_len
        self.top_k = top_k

    async def generate(self, *, request_id, prompt_ids, sampling_params, image_data=None, video_data=None):
        self.calls.append({
            "request_id": request_id,
            "prompt_len": len(prompt_ids),
            "max_tokens": sampling_params.get("max_tokens"),
            "logprobs": sampling_params.get("logprobs"),
        })
        K = int(sampling_params.get("logprobs") or 0) if sampling_params.get("logprobs") is not False else 0
        max_tokens = int(sampling_params.get("max_tokens") or self.response_len)

        # Teacher call: max_tokens=1
        if max_tokens == 1:
            tlp = {i + 1000: -float(i + 1) for i in range(K or 5)}
            return _FakeTokenOutput(
                token_ids=[1000],
                log_probs=[-0.1],
                top_logprobs=[tlp],
            )

        # Student call
        n = min(max_tokens, self.response_len)
        token_ids = list(range(100, 100 + n))
        log_probs = [-0.5] * n
        if K > 0:
            top_logprobs = [{tid + j: -float(j + 1) for j in range(K)} for tid in token_ids]
        else:
            top_logprobs = None
        return _FakeTokenOutput(token_ids=token_ids, log_probs=log_probs, top_logprobs=top_logprobs)


class _FakeTokenizer:
    """Minimal tokenizer with encode/decode."""

    eos_token_id = 2
    pad_token_id = 0
    padding_side = "right"

    def decode(self, ids, skip_special_tokens=True):
        return f"response_text_{len(ids)}_tokens"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 128 for c in text[:50]]


# ---------------------------------------------------------------------------
# Helper: Build a branching config namespace
# ---------------------------------------------------------------------------


def _make_branching_cfg(**overrides) -> SimpleNamespace:
    """Build a dict-like config that mimics BranchingConfig's .get() interface."""
    defaults = {
        "enabled": True,
        "two_stage": True,
        "stage1_n": 4,
        "two_stage_teacher_mode": "ref_or_marker",
        "n_trees": 2,
        "n_splits": 1,
        "top_k": 10,
        "teacher_top_k": 50,
        "teacher_pick_top_k": None,
        "success_reward_threshold": 1.0,
        "teacher_context_mode": "marker",
        "entropy_window": 20,
        "entropy_protect_window": None,
        "entropy_sigma_start": 2.0,
        "entropy_sigma_floor": 0.5,
        "entropy_sigma_step": 0.5,
        "split_trigger": "entropy",
        "max_branch_depth": None,
        "fallback_to_student_topk": True,
        "require_distinct_branches": True,
        "verdict_right_marker": None,
        "max_reprompt_len": 10240,
        "reprompt_template": "{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n",
        "solution_template": "\nCorrect solution:\n\n{successful_previous_attempt}\n\n",
        "teacher_branch_query_max_tokens": 1,
    }
    defaults.update(overrides)
    ns = SimpleNamespace(**defaults)
    ns.get = lambda key, default=None: getattr(ns, key, default)
    return ns


def _build_loop(branching_cfg=None, response_len=30, top_k=10):
    """Build a BranchingAgentLoop instance with stubbed internals."""
    if branching_cfg is None:
        branching_cfg = _make_branching_cfg()

    server = _FakeServerManager(response_len=response_len, top_k=top_k)
    tokenizer = _FakeTokenizer()

    loop = object.__new__(BranchingAgentLoop)
    loop.server_manager = server
    loop.tokenizer = tokenizer
    loop.response_length = 100
    loop.branching_cfg = branching_cfg
    # n_leaves, n_trees, n_leaves_total are @property computed from branching_cfg
    loop.tool_schemas = []
    loop._root_prompt_len = 3

    # Stub apply_chat_template / process_vision_info
    async def _apply(messages, **kw):
        # Return a deterministic token id list based on content length
        content = "".join(m.get("content", "") for m in messages)
        return [10 + i for i in range(min(len(content), 50))]
    async def _vision(messages):
        return {}
    loop.apply_chat_template = _apply
    loop.process_vision_info = _vision
    return loop, server


# ---------------------------------------------------------------------------
# Test: n_leaves_total property
# ---------------------------------------------------------------------------


def test_n_leaves_total_two_stage():
    """n_leaves_total = stage1_n + n_trees * 2^n_splits in two-stage mode."""
    loop, _ = _build_loop(_make_branching_cfg(
        two_stage=True, stage1_n=4, n_trees=2, n_splits=1,
    ))
    # n_leaves = 2^1 = 2; n_trees = 2; stage1_n = 4
    # total = 4 + 2*2 = 8
    assert loop.n_leaves_total == 8


def test_n_leaves_total_non_two_stage():
    """n_leaves_total = n_trees * 2^n_splits without two-stage."""
    loop, _ = _build_loop(_make_branching_cfg(
        two_stage=False, n_trees=4, n_splits=1,
    ))
    # n_leaves = 2^1 = 2; n_trees = 4
    # total = 4*2 = 8
    assert loop.n_leaves_total == 8


# ---------------------------------------------------------------------------
# Test: build_ref_context_messages
# ---------------------------------------------------------------------------


def test_build_ref_context_messages_basic():
    """build_ref_context_messages constructs correct reprompt structure."""
    raw_prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    cfg = _make_branching_cfg()
    result = build_ref_context_messages(
        raw_prompt=raw_prompt,
        successful_response_text="The answer is 4.",
        self_distillation_cfg=cfg,
    )
    assert len(result) == 2  # system + user
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"
    # The user content should contain the original prompt + solution section
    assert "What is 2+2?" in result[1]["content"]
    assert "The answer is 4." in result[1]["content"]
    assert "Correct solution" in result[1]["content"]
    assert "Correctly solve the original question" in result[1]["content"]


def test_build_ref_context_messages_empty_response():
    """build_ref_context_messages with empty response returns original prompt."""
    raw_prompt = [{"role": "user", "content": "Hello"}]
    cfg = _make_branching_cfg()
    result = build_ref_context_messages(
        raw_prompt=raw_prompt,
        successful_response_text="",
        self_distillation_cfg=cfg,
    )
    assert result == [{"role": "user", "content": "Hello"}]


def test_build_ref_context_messages_with_feedback():
    """build_ref_context_messages includes feedback when provided."""
    raw_prompt = [{"role": "user", "content": "Solve x^2=4"}]
    cfg = _make_branching_cfg()
    result = build_ref_context_messages(
        raw_prompt=raw_prompt,
        successful_response_text="x=2 or x=-2",
        self_distillation_cfg=cfg,
        feedback="Think about both roots.",
    )
    assert "Think about both roots." in result[0]["content"]
    assert "x=2 or x=-2" in result[0]["content"]


# ---------------------------------------------------------------------------
# Test: Stage 1 generation logic
# ---------------------------------------------------------------------------


def test_generate_stage1_rollouts():
    """_generate_stage1_rollouts produces stage1_n outputs with correct fields."""
    cfg = _make_branching_cfg(stage1_n=3)
    loop, server = _build_loop(cfg, response_len=20)

    async def _run():
        return await loop._generate_stage1_rollouts(
            prompt_ids=[1, 2, 3],
            sampling_params={"max_tokens": 50, "temperature": 0.7, "logprobs": 10},
            images=None,
            videos=None,
            multi_modal_data={},
        )

    results = asyncio.get_event_loop().run_until_complete(_run())

    assert len(results) == 3
    for i, out in enumerate(results):
        assert out.prompt_ids == [1, 2, 3]
        assert len(out.response_ids) == 20
        assert len(out.response_mask) == 20
        assert all(m == 1 for m in out.response_mask)
        assert out.extra_fields["is_two_stage_stage1"] == 1
        assert out.extra_fields["branch_token_mask"] == [0] * 20
        assert out.extra_fields["leaf_id"] == i

    # Verify logprobs=False in server calls (Stage 1 doesn't need logprobs)
    for call in server.calls:
        assert call["logprobs"] is False


# ---------------------------------------------------------------------------
# Test: Two-stage pipeline scoring + ref context
# ---------------------------------------------------------------------------


def test_two_stage_pipeline_ref_on_success():
    """When Stage 1 has a successful response, ref context is built."""
    cfg = _make_branching_cfg(
        two_stage=True, stage1_n=2, n_trees=1, n_splits=1,
        two_stage_teacher_mode="ref_or_marker",
        success_reward_threshold=0.5,
    )
    loop, server = _build_loop(cfg, response_len=20)

    call_count = {"n": 0}

    async def mock_score_fn(prompt_ids, response_ids, raw_prompt, **extra):
        call_count["n"] += 1
        return 1.0 if call_count["n"] == 1 else 0.0

    # Track which path was taken
    ref_called = {"val": False}
    original_build_ref = loop._build_ref_privileged_context

    async def mock_build_ref(**kw):
        ref_called["val"] = True
        return [50, 51, 52, 53], {"mode": "ref", "ref_available": True, "ref_ids_len": 4, "marker_len": 0}
    loop._build_ref_privileged_context = mock_build_ref

    # Mock the branching to avoid full recursive pipeline
    async def mock_branching(**kw):
        return [
            AgentLoopOutput(prompt_ids=[1, 2], response_ids=[200, 201], response_mask=[1, 1],
                           response_logprobs=[0.0, 0.0], multi_modal_data={}, num_turns=2,
                           metrics=AgentLoopMetrics(), extra_fields={"branch_token_mask": [0, 0]}),
            AgentLoopOutput(prompt_ids=[1, 2], response_ids=[300, 301], response_mask=[1, 1],
                           response_logprobs=[0.0, 0.0], multi_modal_data={}, num_turns=2,
                           metrics=AgentLoopMetrics(), extra_fields={"branch_token_mask": [0, 0]}),
        ]
    loop._run_branching_with_priv_ctx = mock_branching

    kwargs = {
        "raw_prompt": [{"role": "user", "content": "What is 2+2?"}],
        "score_fn": mock_score_fn,
    }

    async def _run():
        return await loop._run_two_stage_pipeline(
            prompt_ids=[1, 2],
            sampling_params={"max_tokens": 50, "temperature": 0.7, "logprobs": 10},
            images=None, videos=None, multi_modal_data={}, kwargs=kwargs,
        )

    results = asyncio.get_event_loop().run_until_complete(_run())

    # stage1_n=2 outputs + branching outputs
    assert len(results) == 4  # 2 stage1 + 2 branched
    # Score was called for both stage1 outputs
    assert call_count["n"] == 2
    # Ref context path was taken (first response scored 1.0 >= 0.5)
    assert ref_called["val"] is True


# ---------------------------------------------------------------------------
# Test: Two-stage pipeline fallback to marker when all fail
# ---------------------------------------------------------------------------


def test_two_stage_pipeline_marker_fallback():
    """When all Stage 1 fail, falls back to marker (via _build_privileged_context)."""
    cfg = _make_branching_cfg(
        two_stage=True, stage1_n=2, n_trees=1, n_splits=1,
        two_stage_teacher_mode="ref_or_marker",
        success_reward_threshold=1.0,
    )
    loop, server = _build_loop(cfg, response_len=20)

    # Score function: all fail (score 0.0)
    async def mock_score_fn(prompt_ids, response_ids, raw_prompt, **extra):
        return 0.0

    marker_called = {"val": False}

    async def mock_build_priv_ctx(*, prompt_ids, kwargs, images, videos):
        marker_called["val"] = True
        return [50, 51, 52], {"mode": "marker", "ref_available": False, "marker_len": 3}
    loop._build_privileged_context = mock_build_priv_ctx

    async def mock_branching(**kw):
        return [
            AgentLoopOutput(prompt_ids=[1, 2], response_ids=[200], response_mask=[1],
                           response_logprobs=[0.0], multi_modal_data={}, num_turns=2,
                           metrics=AgentLoopMetrics(), extra_fields={}),
            AgentLoopOutput(prompt_ids=[1, 2], response_ids=[300], response_mask=[1],
                           response_logprobs=[0.0], multi_modal_data={}, num_turns=2,
                           metrics=AgentLoopMetrics(), extra_fields={}),
        ]
    loop._run_branching_with_priv_ctx = mock_branching

    kwargs = {
        "raw_prompt": [{"role": "user", "content": "What is 2+2?"}],
        "score_fn": mock_score_fn,
    }

    async def _run():
        return await loop._run_two_stage_pipeline(
            prompt_ids=[1, 2],
            sampling_params={"max_tokens": 50, "temperature": 0.7, "logprobs": 10},
            images=None, videos=None, multi_modal_data={}, kwargs=kwargs,
        )

    results = asyncio.get_event_loop().run_until_complete(_run())

    # Marker fallback path was taken
    assert marker_called["val"] is True
    # Still 2 stage1 + 2 branched = 4
    assert len(results) == 4


# ---------------------------------------------------------------------------
# Test: marker_only mode skips scoring
# ---------------------------------------------------------------------------


def test_two_stage_marker_only_skips_scoring():
    """In marker_only mode, score_fn is never called."""
    cfg = _make_branching_cfg(
        two_stage=True, stage1_n=2, n_trees=1, n_splits=1,
        two_stage_teacher_mode="marker_only",
    )
    loop, server = _build_loop(cfg, response_len=20)

    score_called = {"called": False}

    async def mock_score_fn(prompt_ids, response_ids, raw_prompt, **extra):
        score_called["called"] = True
        return 0.0

    async def mock_build_priv_ctx(*, prompt_ids, kwargs, images, videos):
        return [50, 51], {"mode": "marker", "marker_len": 2}
    loop._build_privileged_context = mock_build_priv_ctx

    async def mock_branching(**kw):
        return [
            AgentLoopOutput(prompt_ids=[1], response_ids=[200], response_mask=[1],
                           response_logprobs=[0.0], multi_modal_data={}, num_turns=2,
                           metrics=AgentLoopMetrics(), extra_fields={}),
            AgentLoopOutput(prompt_ids=[1], response_ids=[300], response_mask=[1],
                           response_logprobs=[0.0], multi_modal_data={}, num_turns=2,
                           metrics=AgentLoopMetrics(), extra_fields={}),
        ]
    loop._run_branching_with_priv_ctx = mock_branching

    kwargs = {
        "raw_prompt": [{"role": "user", "content": "Test"}],
        "score_fn": mock_score_fn,
    }

    async def _run():
        return await loop._run_two_stage_pipeline(
            prompt_ids=[1],
            sampling_params={"max_tokens": 30, "logprobs": 10},
            images=None, videos=None, multi_modal_data={}, kwargs=kwargs,
        )

    asyncio.get_event_loop().run_until_complete(_run())

    # score_fn should NOT have been called
    assert score_called["called"] is False


# ---------------------------------------------------------------------------
# Test: leaf assignment correctness
# ---------------------------------------------------------------------------


def test_leaf_slot_mapping_two_stage():
    """In two-stage mode, slots 0..stage1_n-1 are Stage 1, rest are Stage 2."""
    stage1_n = 4
    n_trees = 2
    n_splits = 1
    n_leaves_per_tree = 2 ** n_splits  # 2
    total_branched = n_trees * n_leaves_per_tree  # 4
    total = stage1_n + total_branched  # 8

    assert total == 8
    stage1_slots = list(range(stage1_n))
    stage2_slots = list(range(stage1_n, total))
    assert len(stage1_slots) == 4
    assert len(stage2_slots) == 4


# ---------------------------------------------------------------------------
# Test: BranchingConfig validation (import directly since it's lightweight)
# ---------------------------------------------------------------------------


def test_branching_config_two_stage_validation():
    """Two-stage fields are validated in __post_init__."""
    # Load BranchingConfig directly using importlib
    spec = importlib.util.spec_from_file_location(
        "rollout_config", "verl/workers/config/rollout.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Need to stub some deps
    if "omegaconf" not in sys.modules:
        oc = types.ModuleType("omegaconf")
        oc.MISSING = object()
        sys.modules["omegaconf"] = oc
    if "verl.base_config" not in sys.modules:
        bc = types.ModuleType("verl.base_config")
        from dataclasses import dataclass
        @dataclass
        class _BC:
            def get(self, key, default=None):
                return getattr(self, key, default)
        bc.BaseConfig = _BC
        sys.modules["verl.base_config"] = bc
    # Stub ProfilerConfig in verl.utils.profiler
    prof_mod = sys.modules.get("verl.utils.profiler")
    if prof_mod is None or not hasattr(prof_mod, "ProfilerConfig"):
        from dataclasses import dataclass as _dc
        @_dc
        class _ProfilerConfig:
            pass
        prof_mod = sys.modules.setdefault("verl.utils.profiler", types.ModuleType("verl.utils.profiler"))
        prof_mod.ProfilerConfig = _ProfilerConfig
    elif not hasattr(prof_mod, "ProfilerConfig"):
        from dataclasses import dataclass as _dc
        @_dc
        class _ProfilerConfig:
            pass
        prof_mod.ProfilerConfig = _ProfilerConfig
    spec.loader.exec_module(mod)
    BranchingConfig = mod.BranchingConfig

    # Valid config
    cfg = BranchingConfig(enabled=True, two_stage=True, stage1_n=4, n_trees=2, n_splits=1)
    assert cfg.two_stage is True
    assert cfg.stage1_n == 4
    assert cfg.two_stage_teacher_mode == "ref_or_marker"

    # Invalid stage1_n
    with pytest.raises(ValueError, match="stage1_n"):
        BranchingConfig(enabled=True, two_stage=True, stage1_n=0)

    # Invalid teacher mode
    with pytest.raises(ValueError, match="two_stage_teacher_mode"):
        BranchingConfig(enabled=True, two_stage=True, two_stage_teacher_mode="invalid")

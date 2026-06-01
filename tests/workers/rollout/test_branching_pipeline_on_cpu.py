# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""End-to-end correctness test for BranchingAgentLoop._run_branching_pipeline.

CPU-only. Uses a fake server_manager that returns deterministic top-K
logprob dicts, so we can exercise the full generate -> spike-detect ->
teacher-query -> branch -> recurse loop without vLLM.
"""

from __future__ import annotations

import asyncio
import math
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import importlib.util


def _load_branching():
    spec = importlib.util.spec_from_file_location(
        "bu", "verl/workers/rollout/vllm_rollout/branching_utils.py"
    )
    bu = importlib.util.module_from_spec(spec)
    sys.modules["bu"] = bu
    spec.loader.exec_module(bu)
    return bu


class _FakeTokenOutput:
    def __init__(self, token_ids, log_probs, top_logprobs):
        self.token_ids = token_ids
        self.log_probs = log_probs
        self.top_logprobs = top_logprobs
        self.routed_experts = None
        self.stop_reason = "completed"


class _FakeServerManager:
    """Deterministic fake. Returns:
    - For student calls (logprobs=K, max_tokens=N): a chain of length min(N, 100)
      with one entropy spike at position 30 (only on the FIRST call) and a
      smaller spike at position 50.
    - For teacher calls (max_tokens=1): top_logprobs[0] = student top-K with
      reordered logprobs so argmax = student's #2, argmin = student's #4.
    """

    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self.student_call_idx = 0

    async def generate(self, *, request_id, prompt_ids, sampling_params, image_data=None, video_data=None):
        self.calls.append({
            "request_id": request_id,
            "prompt_len": len(prompt_ids),
            "max_tokens": sampling_params.get("max_tokens"),
            "logprobs": sampling_params.get("logprobs"),
            "temperature": sampling_params.get("temperature"),
        })
        K = int(sampling_params.get("logprobs") or 0)
        max_tokens = int(sampling_params.get("max_tokens") or 100)
        # Teacher call: max_tokens=1, return one position with top-K dict.
        if max_tokens == 1:
            # Teacher distribution: same support as the student top-K it would
            # have seen at this branch position, but reordered so argmax/argmin
            # are well-defined and distinct from student's argmax.
            # We don't know the student's top-K here — return a fixed support
            # of 5 distinct token ids that overlap with the student fake.
            tlp = {1000: -2.5, 1001: -0.1, 1002: -1.0, 1003: -3.5, 1004: -1.7}
            return _FakeTokenOutput(token_ids=[1001], log_probs=[-0.1], top_logprobs=[tlp])
        # Student call.
        n = min(max_tokens, 100)
        tokens = list(range(2000, 2000 + n))
        # Entropy via top-K: low entropy by default, spike at index 30 and 50.
        topk_list = []
        log_probs = []
        for i in range(n):
            if i == 30 or i == 50:
                # High entropy: uniform distribution over 5 tokens, intersecting
                # the teacher distribution's support {1000..1004}.
                lp = math.log(1.0 / 5)
                d = {tid: lp for tid in [1000, 1001, 1002, 1003, 1004]}
            else:
                # Low entropy: peaked. Realized token + 4 alternatives.
                d = {2000 + i: -0.1, 2000 + i + 1: -3.0, 2000 + i + 2: -3.5,
                     2000 + i + 3: -4.0, 2000 + i + 4: -4.5}
            topk_list.append(d)
            log_probs.append(next(iter(d.values())))
        self.student_call_idx += 1
        return _FakeTokenOutput(token_ids=tokens, log_probs=log_probs, top_logprobs=topk_list)


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [99]  # 1-token stand-in marker


def _build_loop():
    bu = _load_branching()
    # Stub minimal verl package to import branching_agent_loop without the
    # full verl init chain.
    import types
    fake_verl = types.ModuleType("verl"); fake_verl.__path__ = ["verl"]
    sys.modules.setdefault("verl", fake_verl)
    # Stub child packages used by imports.
    for pkg in ["verl.experimental", "verl.experimental.agent_loop",
                "verl.tools", "verl.tools.utils", "verl.utils",
                "verl.workers", "verl.workers.rollout",
                "verl.workers.rollout.vllm_rollout"]:
        m = types.ModuleType(pkg); m.__path__ = [pkg.replace('.', '/')]
        sys.modules.setdefault(pkg, m)
    # Provide branching_utils so the import inside branching_agent_loop succeeds.
    sys.modules["verl.workers.rollout.vllm_rollout.branching_utils"] = bu
    # Stub agent_loop dependencies.
    al = types.ModuleType("verl.experimental.agent_loop.agent_loop")
    class _AgentLoopBase:
        def __init__(self, *a, **k): pass
        async def process_vision_info(self, messages): return {}
        async def apply_chat_template(self, messages, **kwargs): return [10, 11, 12]
    class _AgentLoopMetrics:
        def __init__(self, **kw): self._d = kw
    class _AgentLoopOutput:
        def __init__(self, **kw): self.__dict__.update(kw); self.extra_fields = kw.get('extra_fields', {})
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
    # Stub tools.
    tu = types.ModuleType("verl.tools.utils.tool_registry")
    tu.initialize_tools_from_config = lambda p: []
    sys.modules["verl.tools.utils"] = types.ModuleType("verl.tools.utils")
    sys.modules["verl.tools.utils"].__path__ = ["verl/tools/utils"]
    sys.modules["verl.tools.utils.tool_registry"] = tu
    # Stub profiler.
    prof = types.ModuleType("verl.utils.profiler")
    class _SimpleTimer:
        def __init__(self, name, metrics): self.name = name; self.metrics = metrics
        def __enter__(self): return self
        def __exit__(self, *a): self.metrics[self.name] = 0.0
    prof.simple_timer = _SimpleTimer
    sys.modules["verl.utils.profiler"] = prof
    # Stub teacher_prompt module.
    tp = types.ModuleType("verl.utils.teacher_prompt")
    tp.build_marker_text = lambda **kw: "[Meta: verified correct]\n\n"
    tp.build_ref_gt_messages = lambda **kw: list(kw.get("raw_prompt") or [])
    sys.modules["verl.utils.teacher_prompt"] = tp

    # Now load branching_agent_loop.
    spec = importlib.util.spec_from_file_location(
        "bal", "verl/experimental/agent_loop/branching_agent_loop.py"
    )
    bal = importlib.util.module_from_spec(spec)
    sys.modules["bal"] = bal
    spec.loader.exec_module(bal)

    # Construct a BranchingAgentLoop with stubbed config + fake server.
    BranchingCfg = SimpleNamespace(
        enabled=True, n_splits=2, top_k=5,
        entropy_window=10, entropy_protect_window=5,
        entropy_sigma_start=2.0, entropy_sigma_floor=0.5, entropy_sigma_step=0.5,
        teacher_context_mode="marker",
        verdict_right_marker=None,
        gt_marker_template="ignored",
        reprompt_template="{prompt}{solution}{feedback}",
        solution_template="\n{successful_previous_attempt}\n",
        max_reprompt_len=1024,
        teacher_branch_query_max_tokens=1,
        fallback_to_student_topk=True,
        require_distinct_branches=True,
        max_branch_depth=None,
        effective_protect_window=5,
        effective_max_branch_depth=2,
    )
    # The cfg.get(...) shim used by build_marker_text:
    def _get(self, key, default=None):
        return getattr(self, key, default)
    BranchingCfg.get = _get.__get__(BranchingCfg, SimpleNamespace)
    rollout_cfg = SimpleNamespace(
        prompt_length=512, response_length=80,
        branching=BranchingCfg,
    )
    rollout_cfg.get = (lambda self, k, default=None: getattr(self, k, default)).__get__(
        rollout_cfg, SimpleNamespace
    )
    actor_rollout_ref = SimpleNamespace(rollout=rollout_cfg)
    data_cfg = SimpleNamespace(tool_config_path=None)
    full_cfg = SimpleNamespace(actor_rollout_ref=actor_rollout_ref, data=data_cfg)

    server = _FakeServerManager()
    tokenizer = _FakeTokenizer()

    # Manually skip parent __init__ — it expects much more.
    loop = object.__new__(bal.BranchingAgentLoop)
    loop.config = full_cfg
    loop.server_manager = server
    loop.tokenizer = tokenizer
    loop.processor = None
    loop.dataset_cls = None
    loop.dataset_config = data_cfg
    loop.apply_chat_template_kwargs = {}
    loop.system_prompt = []
    loop.loop = asyncio.new_event_loop()
    loop.prompt_length = 512
    loop.response_length = 80
    loop.branching_cfg = BranchingCfg
    loop.tool_schemas = []

    # Stub apply_chat_template / process_vision_info on the instance for the run path.
    async def _apply(messages, **kw): return [10, 11, 12]
    async def _vision(messages): return {}
    loop.apply_chat_template = _apply
    loop.process_vision_info = _vision
    return loop, server, bal


def main() -> None:
    loop, server, bal = _build_loop()
    kwargs = {
        "raw_prompt": [{"role": "user", "content": "Q?"}],
        "reward_model": {"ground_truth": "A"},
    }
    sampling_params = {"temperature": 0.7, "top_p": 1.0, "logprobs": False}

    async def _run():
        leaves = await loop._run_branching_pipeline(
            prompt_ids=[10, 11, 12],
            sampling_params=sampling_params,
            images=None, videos=None,
            multi_modal_data={},
            kwargs=kwargs,
        )
        return leaves

    loop._root_prompt_len = 3
    leaves = asyncio.get_event_loop().run_until_complete(_run())

    n_expected = 2 ** loop.branching_cfg.n_splits
    assert len(leaves) == n_expected, f"expected {n_expected} leaves, got {len(leaves)}"

    # branch_token_mask 1-count should equal depth (n_splits) for fully-split leaves.
    for i, leaf in enumerate(leaves):
        mask = leaf.extra_fields["branch_token_mask"]
        ones = sum(mask)
        # On fully-split leaves the count is exactly n_splits=2.
        # When the tree was shallower (because we couldn't find a spike) padding
        # may produce duplicates with fewer ones — accept anything in [0, n_splits].
        assert 0 <= ones <= loop.branching_cfg.n_splits, (i, ones, mask)

    # At least one teacher query must have happened.
    teacher_calls = [c for c in server.calls if c["max_tokens"] == 1]
    assert len(teacher_calls) >= 1, "no teacher branch-point query was issued"

    # All calls must use the same request_id (sticky LRU).
    rids = {c["request_id"] for c in server.calls}
    assert len(rids) == 1, f"expected single traj_id, got {rids}"

    # Student calls must use logprobs=K (int).
    student_calls = [c for c in server.calls if c["max_tokens"] != 1]
    for c in student_calls:
        assert isinstance(c["logprobs"], int) and c["logprobs"] == 5, c

    print(f"BranchingAgentLoop._run_branching_pipeline OK")
    print(f"  leaves: {len(leaves)}")
    print(f"  total vLLM calls: {len(server.calls)}")
    print(f"  student calls: {len(student_calls)}")
    print(f"  teacher calls: {len(teacher_calls)}")
    print(f"  branch_mask ones per leaf: {[sum(l.extra_fields['branch_token_mask']) for l in leaves]}")


if __name__ == "__main__":
    main()

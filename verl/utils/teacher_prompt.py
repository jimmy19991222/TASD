# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared string-formatting helpers for SDPO teacher prompt construction.

Both the trainer-side teacher prompt builder
(``verl.trainer.ppo.ray_trainer._build_self_distillation_batch_marker`` /
``_build_self_distillation_batch_ref``) and the rollout-time
``verl.experimental.agent_loop.branching_agent_loop.BranchingAgentLoop`` need
to construct privileged-context teacher prompts in three modes:

    - marker    : prepend a static verdict marker between the assistant role-start
                  and the response.
    - gt_marker : same, but with a per-sample marker built from the parquet
                  ``ground_truth`` field (falls back to the static marker when
                  ``ground_truth`` is unavailable).
    - ref_gt    : insert the ``ground_truth`` as the SDPO ``solution`` in the
                  user-side reprompt (no assistant-side marker).

The trainer operates on padded GPU tensors and a full DataProto; the agent
loop operates on per-row token id lists. The two cannot share their tensor
plumbing, but they MUST share the same string-formatting rules — otherwise the
teacher distribution at branch points (rollout) and the teacher distribution
inside the SDPO loss (training) would drift apart, silently invalidating the
on-policy assumption.

This module is the single source of truth for those rules.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from verl.utils.verdict_markers import VERDICT_RIGHT_MARKER, VERDICT_WRONG_MARKER

__all__ = [
    "VERDICT_RIGHT_MARKER",
    "VERDICT_WRONG_MARKER",
    "resolve_right_marker",
    "build_marker_text",
    "build_ref_gt_messages",
]


def resolve_right_marker(self_distillation_cfg) -> str:
    """Resolve the verified-correct marker text from a config object.

    The config may be a ``SelfDistillationConfig`` (dataclass), an OmegaConf
    DictConfig, or a plain dict (e.g. when called from the rollout side).
    Falls back to ``VERDICT_RIGHT_MARKER`` if no override is set.
    """
    override = _safe_get(self_distillation_cfg, "verdict_right_marker", None)
    return override if override else VERDICT_RIGHT_MARKER


def build_marker_text(
    *,
    mode: str,
    ground_truth: Optional[str],
    self_distillation_cfg,
) -> str:
    """Construct the marker text that gets prepended between the assistant
    role-start and the response, for ``marker`` / ``gt_marker`` modes.

    The trailing ``\\n\\n`` is always present so the marker visually separates
    from the response. The trainer used to add this manually after templating;
    centralising it here ensures the rollout side matches.

    Args:
        mode: ``"marker"`` or ``"gt_marker"``.
        ground_truth: per-sample ground truth (only consulted for ``gt_marker``).
            On None / empty string we silently fall back to the static right
            marker — same behaviour as the trainer-side builder before this
            refactor.
        self_distillation_cfg: any object that supports ``.get("...")`` /
            attribute access. Reads ``verdict_right_marker`` and
            ``gt_marker_template``.
    """
    if mode not in ("marker", "gt_marker"):
        raise ValueError(f"build_marker_text expected mode in {{marker, gt_marker}}, got {mode!r}")

    right_marker = resolve_right_marker(self_distillation_cfg)
    if mode == "marker" or not ground_truth:
        text = right_marker
    else:
        template = _safe_get(self_distillation_cfg, "gt_marker_template", None)
        if not template:
            text = right_marker
        else:
            text = template.format(ground_truth=ground_truth)

    if not text.endswith("\n\n"):
        text = text + "\n\n"
    return text


def build_ref_gt_messages(
    *,
    raw_prompt: Sequence[dict],
    ground_truth: Optional[str],
    self_distillation_cfg,
    feedback: str = "",
) -> list[dict]:
    """Rebuild a chat-template message list with ``ground_truth`` slotted in
    as the SDPO ``solution`` section of the reprompt.

    Mirrors ``_build_self_distillation_batch_ref`` user-message construction
    (verl/trainer/ppo/ray_trainer.py:768-797). Returns a fresh list; safe to
    pass to ``tokenizer.apply_chat_template`` or
    ``AgentLoopBase.apply_chat_template``.

    Args:
        raw_prompt: original chat-template messages (system + user); the last
            element must be the user turn (its content gets re-formatted).
        ground_truth: per-sample ground truth; on None falls back to
            ``raw_prompt`` unchanged.
        self_distillation_cfg: reads ``reprompt_template``, ``solution_template``.
        feedback: optional environment feedback string. Defaults to empty.
    """
    if not raw_prompt:
        return []
    if not ground_truth:
        return list(raw_prompt)

    system_messages = list(raw_prompt[:-1])
    last_user = raw_prompt[-1]
    if not isinstance(last_user, dict) or "content" not in last_user:
        return list(raw_prompt)
    prompt_text = last_user["content"]

    solution_template = _safe_get(
        self_distillation_cfg, "solution_template",
        "\nCorrect solution:\n\n{successful_previous_attempt}\n\n",
    )
    feedback_template = _safe_get(
        self_distillation_cfg, "feedback_template",
        "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n\n",
    )
    reprompt_template = _safe_get(
        self_distillation_cfg, "reprompt_template",
        "{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n",
    )

    solution_section = solution_template.format(successful_previous_attempt=ground_truth)
    feedback_section = feedback_template.format(feedback_raw=feedback) if feedback else ""
    reprompt_text = reprompt_template.format(
        prompt=prompt_text,
        solution=solution_section,
        feedback=feedback_section,
    )
    return system_messages + [{"role": "user", "content": reprompt_text}]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _safe_get(obj: Any, key: str, default: Any) -> Any:
    """Get ``key`` from a dataclass, OmegaConf DictConfig, or plain dict."""
    if obj is None:
        return default
    # OmegaConf DictConfig and dict both expose .get
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            val = getter(key, default)
            return val if val is not None else default
        except Exception:  # noqa: BLE001
            pass
    # Dataclass attribute access fallback.
    val = getattr(obj, key, default)
    return val if val is not None else default

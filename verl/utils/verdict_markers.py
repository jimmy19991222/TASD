"""Shared verdict-marker constants for verifier-conditioned self-distillation.

These are the two contrastive prefixes prepended to the teacher prompt to elicit
the model's right- and wrong-conditioned token distributions (see
research/reward_bayes_distillation_v2.md and research/vc_opsd_paper_draft_zh.md).
Centralised here so that the offline calibration diagnostic and the training
pipeline are guaranteed to use the same wording.
"""

VERDICT_RIGHT_MARKER = (
    "[Meta: the assistant response below is verified to correctly answer the question.]"
)
VERDICT_WRONG_MARKER = (
    "[Meta: the assistant response below is verified to incorrectly answer the question.]"
)


def get_verdict_marker(is_right: bool) -> str:
    return VERDICT_RIGHT_MARKER if is_right else VERDICT_WRONG_MARKER

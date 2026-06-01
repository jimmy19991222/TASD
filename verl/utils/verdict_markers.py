"""Shared verdict-marker constants for teacher_context_mode ∈ {marker, gt_marker}.

These are the two contrastive prefixes prepended to the teacher's view of the
assistant response, used by the marker/gt_marker teacher contexts in SDPO and
by the upcoming teacher-guided branching rollout (see
research/teacher_branching_rollout.md).
"""

VERDICT_RIGHT_MARKER = (
    "[Meta: the assistant response below is verified to correctly answer the question.]"
)
VERDICT_WRONG_MARKER = (
    "[Meta: the assistant response below is verified to incorrectly answer the question.]"
)


def get_verdict_marker(is_right: bool) -> str:
    return VERDICT_RIGHT_MARKER if is_right else VERDICT_WRONG_MARKER

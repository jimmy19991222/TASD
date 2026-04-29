import re
import json
from collections import Counter


def extract_actions(text: str) -> list[str]:
    """Extract all action names after 'Action:' occurrences."""
    actions = re.findall(r'Action:\s*(\w+)', text)
    return actions


def extract_action_inputs(text: str) -> dict:
    """Extract and merge all JSON blocks following 'Action Input:'."""
    json_blocks = re.findall(r'Action Input:\s*({.*?})', text, re.DOTALL)
    
    combined_dict = {}
    for block in json_blocks:
        try:
            parsed = json.loads(block)
            combined_dict.update(parsed)
        except json.JSONDecodeError:
            pass
    
    return combined_dict


def merge_action_inputs(action_inputs_list: list[dict]) -> dict:
    """Merge a list of action input dicts into a single dict."""
    combined = {}
    for d in action_inputs_list:
        if d:
            combined.update(d)
    return combined


def diagnose_format_error(text: str) -> tuple[bool, str]:
    """
    Diagnose format errors in tooluse response.
    
    Returns:
        (is_correct: bool, error_msg: str)
        is_correct=True  -> error_msg is empty
        is_correct=False -> error_msg describes the specific format error
    """
    has_action = re.search(r'Action:', text) is not None
    has_action_input = re.search(r'Action Input:', text) is not None
    
    # 1. Missing fields
    if not has_action and not has_action_input:
        return False, "Format error: missing both 'Action:' and 'Action Input:' fields"
    
    if has_action and not has_action_input:
        return False, "Format error: missing 'Action Input:' field"
    
    if not has_action and has_action_input:
        return False, "Format error: missing 'Action:' field"
    
    # 2. Has both fields, check JSON parsing
    json_blocks = re.findall(r'Action Input:\s*({.*?})', text, re.DOTALL)
    if not json_blocks:
        return False, "Format error: 'Action Input:' has no JSON content"
    
    for block in json_blocks:
        try:
            json.loads(block)
        except json.JSONDecodeError as e:
            return False, f"Format error: JSON parse error in Action Input ({str(e)})"
    
    return True, ""


def is_correct_format(text: str) -> bool:
    """Check if the text contains the expected Action/Action Input format."""
    is_correct, _ = diagnose_format_error(text)
    return is_correct


def compute_score(solution: str, ground_truth: str) -> dict:
    """
    Compute score for tooluse task.
    
    Args:
        solution: The model's response text
        ground_truth: JSON string containing list of dicts with 'Action' and 'Action_Input' keys
                      e.g., '[{"Action": "search", "Action_Input": "{\"query\": \"test\"}"}]'
    
    Returns:
        dict with score, acc, pred, incorrect_format, feedback
    """
    # Parse ground truth
    try:
        gt_list = json.loads(ground_truth)
    except json.JSONDecodeError:
        # If ground_truth is already a list (passed directly), handle that case
        if isinstance(ground_truth, list):
            gt_list = ground_truth
        else:
            return {
                "score": 0.0,
                "acc": 0.0,
                "pred": "",
                "incorrect_format": 1,
                "feedback": "Failed to parse ground truth JSON",
            }
    
    # Extract ground truth actions and action inputs
    gt_actions = [item['Action'] for item in gt_list]
    gt_action_inputs_list = []
    for item in gt_list:
        try:
            parsed_input = json.loads(item['Action_Input']) if isinstance(item['Action_Input'], str) else item['Action_Input']
            gt_action_inputs_list.append(parsed_input)
        except (json.JSONDecodeError, KeyError):
            gt_action_inputs_list.append({})
    gt_action_inputs = merge_action_inputs(gt_action_inputs_list)
    
    # Extract predicted actions and action inputs from solution
    pred_actions = extract_actions(solution)
    pred_action_inputs = extract_action_inputs(solution)
    
    # Check correctness
    actions_correct = Counter(pred_actions) == Counter(gt_actions)
    action_inputs_correct = pred_action_inputs == gt_action_inputs
    
    # Both must be correct for full score
    is_correct = actions_correct and action_inputs_correct
    reward = 1.0 if is_correct else 0.0
    
    # Check format with detailed diagnosis
    correct_format, format_error_msg = diagnose_format_error(solution)
    
    # Build prediction string for logging
    prediction = f"Actions: {pred_actions}, Inputs: {pred_action_inputs}"
    
    # Build detailed feedback
    feedback_parts = []
    
    # 1. Format error feedback
    if not correct_format:
        feedback_parts.append(format_error_msg)
    
    # 2. Action error feedback
    if not actions_correct:
        if not pred_actions:
            feedback_parts.append(f"Action error: no action detected, expected {gt_actions}")
        else:
            # Identify specific action mismatches
            pred_set = set(pred_actions)
            gt_set = set(gt_actions)
            extra_actions = list(pred_set - gt_set)
            missing_actions = list(gt_set - pred_set)
            
            if extra_actions and missing_actions:
                feedback_parts.append(
                    f"Action error: should call {gt_actions}, but called {pred_actions} "
                    f"(wrong action: used {extra_actions} instead of {missing_actions})"
                )
            elif extra_actions:
                feedback_parts.append(
                    f"Action error: should call {gt_actions}, but called {pred_actions} "
                    f"(extra action: {extra_actions})"
                )
            elif missing_actions:
                feedback_parts.append(
                    f"Action error: should call {gt_actions}, but called {pred_actions} "
                    f"(missing action: {missing_actions})"
                )
            else:
                # Same actions but different counts (multiset mismatch)
                feedback_parts.append(
                    f"Action error: should call {gt_actions}, but called {pred_actions} "
                    f"(action count mismatch)"
                )
    
    # 3. Action Input error feedback
    if not action_inputs_correct and pred_actions:
        # Only report input errors if actions are present (otherwise it's covered above)
        if not pred_action_inputs:
            feedback_parts.append(f"Input error: no action input detected, expected {gt_action_inputs}")
        else:
            # Identify specific key/value mismatches
            pred_keys = set(pred_action_inputs.keys())
            gt_keys = set(gt_action_inputs.keys())
            extra_keys = list(pred_keys - gt_keys)
            missing_keys = list(gt_keys - pred_keys)
            wrong_values = []
            for key in pred_keys & gt_keys:
                if pred_action_inputs[key] != gt_action_inputs[key]:
                    wrong_values.append(
                        f"{key} should be {gt_action_inputs[key]}, got {pred_action_inputs[key]}"
                    )
            
            parts = []
            if missing_keys:
                parts.append(f"missing keys: {missing_keys}")
            if extra_keys:
                parts.append(f"extra keys: {extra_keys}")
            if wrong_values:
                parts.append("; ".join(wrong_values))
            
            if parts:
                feedback_parts.append(
                    f"Input error: expected {gt_action_inputs}, but got {pred_action_inputs} "
                    f"({'; '.join(parts)})"
                )
            else:
                feedback_parts.append(
                    f"Input error: expected {gt_action_inputs}, but got {pred_action_inputs}"
                )

    if len(feedback_parts) == 0:
        feedback = ""  # no feedback means correct
        format_error_type = "none"
    else:
        feedback = "; ".join(feedback_parts)
        # Categorize format error for monitoring
        if not correct_format:
            if "missing both" in format_error_msg:
                format_error_type = "missing_both"
            elif "missing 'Action:'" in format_error_msg:
                format_error_type = "missing_action"
            elif "missing 'Action Input:'" in format_error_msg:
                format_error_type = "missing_action_input"
            elif "no JSON content" in format_error_msg:
                format_error_type = "empty_json"
            elif "JSON parse error" in format_error_msg:
                format_error_type = "json_parse_error"
            else:
                format_error_type = "other_format"
        else:
            format_error_type = "none"
    
    return {
        "score": reward,
        "acc": reward,
        "pred": prediction,
        "incorrect_format": 0 if correct_format else 1,
        "feedback": feedback,
        "format_error_type": format_error_type,
    }

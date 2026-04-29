import re


def extract_xml_answer(text: str) -> str:
    """Extract answer from XML-formatted text."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def is_correct_format(text: str) -> bool:
    """
    Check if the text is in the correct XML format.

    The text should contain at the end of the text:
    <answer>
    (A|B|C|D)
    </answer>
    """
    pattern = r"<answer>\s*(A|B|C|D)\s*</answer>$"
    return re.search(pattern, text) is not None

def compute_score(solution: str, ground_truth: str) -> dict:
    multiple_choice_answer = extract_xml_answer(solution)

    reward = float(multiple_choice_answer == ground_truth)
    correct_format = is_correct_format(solution)

    # Build detailed feedback
    feedback_parts = []
    if not correct_format:
        feedback_parts.append(
            "Format error: response does not contain valid `<answer>...</answer>` tags"
        )
    elif reward < 1.0:
        # Format is correct but answer is wrong
        feedback_parts.append(
            f"Answer error: predicted {multiple_choice_answer}, expected {ground_truth}"
        )

    if len(feedback_parts) == 0:
        feedback = ""  # correct, no feedback needed
    else:
        feedback = "; ".join(feedback_parts)

    return {
      "score": reward,
      "acc": reward,
      "pred": multiple_choice_answer,
      "incorrect_format": 1 if correct_format else 0,
      "feedback": feedback,
    }

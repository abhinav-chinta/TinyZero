# verl/utils/reward_score/relevance.py
#
# Author: 2025 – your‑name‑here
# Licence: Apache‑2.0  (same as the rest of VERL)

"""
Reward function for binary relevance classification prompts of the form

    <reasoning> ... </reasoning>
    <relevance>0|1</relevance>

The model earns full reward (`score`, default = 1.0) when the value inside
<relevance>…</relevance> matches the integer found in
`ground_truth["ground_truth"]`, a partial formatting reward (`format_score`,
default = 0.0) when the tag is present but wrong, and 0 if the tag cannot be
parsed at all.
"""

import re
from typing import Optional


# ---------- helper --------------------------------------------------------- #
def _strip_leading_assistant(text: str) -> str:
    """Remove leading chat role tokens that sometimes precede the answer."""
    for marker in ("Assistant:", "<|im_start|>assistant"):
        if marker in text:
            text = text.split(marker, 1)[1]
    return text


# ---------- public API ----------------------------------------------------- #
def extract_solution(solution_str: str, method: str = "strict") -> Optional[str]:
    """
    Parameters
    ----------
    solution_str : str
        Raw model output.
    method       : {"strict","flexible"}
        * strict   - require explicit <relevance> tag
        * flexible - fall back to the last literal '0' or '1' in the string

    Returns
    -------
    str | None
        "0" or "1" if found, else None.
    """
    assert method in {"strict", "flexible"}

    cleaned = _strip_leading_assistant(solution_str)

    if method == "strict":
        # tolerant to extra whitespace and case
        m = re.search(r"<\s*relevance\s*>\s*([01])\s*</\s*relevance\s*>",
                      cleaned, flags=re.I | re.S)
        return m.group(1) if m else None

    # ---------- flexible --------------------------------------------------- #
    # Grab the last digit 0/1 in the text (avoids false hits in <reasoning>)
    digits = re.findall(r"[01]", cleaned)
    return digits[-1] if digits else None


def compute_score(solution_str: str,
                  ground_truth: dict,
                  method: str = "strict",
                  format_score: float = 0.0,
                  score: float = 1.0) -> float:
    """
    Parameters
    ----------
    solution_str : str
        Raw model output.
    ground_truth : dict
        Must contain key `"ground_truth"` with int 0 or 1.
    method       : {"strict","flexible"}
    format_score : float
        Reward when tag is well-formed but prediction is wrong.
    score        : float
        Reward for a correct prediction.

    Returns
    -------
    float
        Reward in [0, score].
    """
    target = int(ground_truth["ground_truth"])
    pred = extract_solution(solution_str, method=method)

    if pred is None:           # no parsable answer
        return 0.0
    if int(pred) == target:    # correct
        return float(score)
    return float(format_score)  # wrong but well‑formed
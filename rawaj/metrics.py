"""Decision metrics for a mule-risk score.

The functions in this module operate only on the supplied labels and scores;
they do not read files or access the network.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _validated_inputs(y_true: Sequence, scores: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Return validated one-dimensional label and score arrays."""
    try:
        y = np.asarray(y_true)
        score_array = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true and scores must be one-dimensional numeric sequences") from exc

    if y.ndim != 1 or score_array.ndim != 1:
        raise ValueError("y_true and scores must be one-dimensional")
    if y.size == 0 or score_array.size == 0:
        raise ValueError("y_true and scores must not be empty")
    if y.size != score_array.size:
        raise ValueError("y_true and scores must have the same length")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("every label in y_true must be 0 or 1")
    if np.isnan(score_array).any():
        raise ValueError("scores must not contain NaN")

    # Integer labels make downstream comparisons predictable while accepting
    # valid representations such as 0.0 and 1.0.
    return y.astype(np.int8, copy=False), score_array


def _confusion_from_validated(
    y: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, int]:
    flagged = scores >= threshold
    positive = y == 1
    return {
        "tp": int(np.count_nonzero(flagged & positive)),
        "fp": int(np.count_nonzero(flagged & ~positive)),
        "tn": int(np.count_nonzero(~flagged & ~positive)),
        "fn": int(np.count_nonzero(~flagged & positive)),
    }


def confusion_at(y_true, scores, threshold: float) -> dict:
    """Return {"tp": int, "fp": int, "tn": int, "fn": int} at `threshold`."""
    y, score_array = _validated_inputs(y_true, scores)
    return _confusion_from_validated(y, score_array, threshold)


def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    """Return (precision, recall) at `threshold`."""
    y, score_array = _validated_inputs(y_true, scores)
    counts = _confusion_from_validated(y, score_array, threshold)
    flagged_count = counts["tp"] + counts["fp"]
    positive_count = counts["tp"] + counts["fn"]
    precision = counts["tp"] / flagged_count if flagged_count else 0.0
    recall = counts["tp"] / positive_count if positive_count else 0.0
    return float(precision), float(recall)


def precision_at_k(y_true, scores, k: int) -> float:
    """Share of positives among the k highest scores.

    ValueError if k <= 0; if k > n, use n. Rank by score descending; ties keep their
    original order.
    """
    y, score_array = _validated_inputs(y_true, scores)
    if k <= 0:
        raise ValueError("k must be greater than 0")

    count = min(int(k), len(y))
    # Python's sort is stable, so equal scores retain their original order.
    ranked_indices = sorted(range(len(score_array)), key=lambda index: score_array[index], reverse=True)
    top_indices = ranked_indices[:count]
    return float(np.count_nonzero(y[top_indices] == 1) / count)


def expected_cost(y_true, scores, threshold: float, *, cost_fp: float, cost_fn: float) -> float:
    """fp * cost_fp + fn * cost_fn at `threshold`."""
    y, score_array = _validated_inputs(y_true, scores)
    counts = _confusion_from_validated(y, score_array, threshold)
    return float(counts["fp"] * cost_fp + counts["fn"] * cost_fn)


def best_threshold(y_true, scores, *, cost_fp: float, cost_fn: float) -> float:
    """Threshold with the lowest expected cost.

    Candidates are every distinct score plus math.inf ("flag nothing"). On a tie in cost,
    return the highest threshold (fewer flags for the same cost).
    """
    y, score_array = _validated_inputs(y_true, scores)
    # Descending order plus updating only for a strictly lower cost preserves
    # the highest threshold when multiple candidates have the same cost.
    candidates = sorted(
        {float(value) for value in score_array} | {math.inf}, reverse=True
    )  # math.inf is the specified candidate for flagging nothing.

    best_candidate = candidates[0]
    best_cost = math.inf
    for candidate in candidates:
        counts = _confusion_from_validated(y, score_array, candidate)
        cost = counts["fp"] * cost_fp + counts["fn"] * cost_fn
        if cost < best_cost:
            best_cost = cost
            best_candidate = candidate
    return float(best_candidate)


def threshold_for_precision(y_true, scores, min_precision: float) -> float | None:
    """Lowest distinct score whose precision is >= min_precision, or None if none reaches it."""
    y, score_array = _validated_inputs(y_true, scores)
    for candidate in sorted(set(float(value) for value in score_array)):
        counts = _confusion_from_validated(y, score_array, candidate)
        flagged_count = counts["tp"] + counts["fp"]
        precision = counts["tp"] / flagged_count if flagged_count else 0.0
        if precision >= min_precision:
            return float(candidate)
    return None

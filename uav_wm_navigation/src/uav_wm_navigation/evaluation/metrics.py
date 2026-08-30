from __future__ import annotations

import numpy as np
from statistics import NormalDist


def polyline_length(points: np.ndarray) -> float:
    """Return the travelled length of a finite [N, 3] navigation trace."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if not np.isfinite(values).all():
        raise ValueError("points must be finite")
    if len(values) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum())


def navigation_error(final_position: np.ndarray, goal_position: np.ndarray) -> float:
    """Final Euclidean distance to the navigation goal in metres (NE)."""

    final = np.asarray(final_position, dtype=np.float64)
    goal = np.asarray(goal_position, dtype=np.float64)
    if final.shape != (3,) or goal.shape != (3,):
        raise ValueError("final_position and goal_position must have shape [3]")
    if not np.isfinite(final).all() or not np.isfinite(goal).all():
        raise ValueError("positions must be finite")
    return float(np.linalg.norm(goal - final))


def success_weighted_path_length(
    success: bool,
    path_length_m: float,
    shortest_path_length_m: float,
) -> float:
    """Episode SPL using the standard success-weighted path-efficiency form."""

    travelled = float(path_length_m)
    shortest = float(shortest_path_length_m)
    if travelled < 0.0 or shortest < 0.0:
        raise ValueError("path lengths must be non-negative")
    if not success:
        return 0.0
    if shortest <= 1e-8:
        return 1.0 if travelled <= 1e-8 else 0.0
    return float(shortest / max(shortest, travelled))


def aggregate_navigation_metrics(
    successes: np.ndarray,
    navigation_errors_m: np.ndarray,
    spl_values: np.ndarray,
) -> dict[str, float | int]:
    """Aggregate the three preregistered month-end navigation metrics."""

    success_array = np.asarray(successes, dtype=bool)
    errors = np.asarray(navigation_errors_m, dtype=np.float64)
    spl = np.asarray(spl_values, dtype=np.float64)
    if not (success_array.ndim == errors.ndim == spl.ndim == 1):
        raise ValueError("metric arrays must be one-dimensional")
    if not (len(success_array) == len(errors) == len(spl)):
        raise ValueError("metric arrays must have equal length")
    if not len(success_array):
        raise ValueError("at least one episode is required")
    if not np.isfinite(errors).all() or not np.isfinite(spl).all():
        raise ValueError("navigation metrics must be finite")
    return {
        "episodes": int(len(success_array)),
        "successes": int(success_array.sum()),
        "sr": float(success_array.mean()),
        "ne_m": float(errors.mean()),
        "spl": float(spl.mean()),
    }


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive, negative = scores[labels == 1], scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    return float(((positive[:, None] > negative).mean() + 0.5 * (positive[:, None] == negative).mean()))


def binary_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(np.asarray(scores))[::-1]
    sorted_labels = labels[order]
    positives = max(int(sorted_labels.sum()), 1)
    recall = np.cumsum(sorted_labels) / positives
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((np.asarray(probabilities) - np.asarray(labels)) ** 2))


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lower) & (probabilities < upper if upper < 1.0 else probabilities <= upper)
        if mask.any():
            total += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return float(total)


def pairwise_ranking_accuracy(target_scores: np.ndarray, predicted_scores: np.ndarray, valid_mask: np.ndarray) -> float:
    target_scores, predicted_scores, valid_mask = map(np.asarray, (target_scores, predicted_scores, valid_mask))
    correct = total = 0
    for target, predicted, valid in zip(target_scores, predicted_scores, valid_mask):
        indices = np.flatnonzero(valid)
        for offset, left in enumerate(indices):
            for right in indices[offset + 1:]:
                if np.isclose(target[left], target[right]): continue
                correct += int(np.sign(target[left] - target[right]) == np.sign(predicted[left] - predicted[right]))
                total += 1
    return float(correct / total) if total else float("nan")


def paired_bootstrap_interval(differences: np.ndarray, seed: int = 0, samples: int = 10_000) -> tuple[float, float, float]:
    differences = np.asarray(differences, dtype=np.float64)
    if not len(differences): return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bootstrap = differences[rng.integers(0, len(differences), size=(samples, len(differences)))].mean(axis=1)
    return float(differences.mean()), float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Binomial proportion and Wilson score interval."""

    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    proportion = successes / trials
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return float(proportion), float(center - radius), float(center + radius)


def ndcg(
    target_relevance: np.ndarray,
    predicted_scores: np.ndarray,
    valid_mask: np.ndarray | None = None,
    k: int | None = None,
) -> float:
    """Mean normalized discounted cumulative gain across trajectory batches."""

    targets = np.asarray(target_relevance, dtype=np.float64)
    predictions = np.asarray(predicted_scores, dtype=np.float64)
    if targets.ndim == 1:
        targets, predictions = targets[None, :], predictions[None, :]
    if targets.shape != predictions.shape or targets.ndim != 2:
        raise ValueError("targets and predictions must share shape [batch, candidates]")
    mask = np.ones_like(targets, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if mask.shape != targets.shape:
        raise ValueError("valid_mask must match targets")
    values = []
    for relevance, scores, valid in zip(targets, predictions, mask):
        relevance = np.maximum(relevance[valid], 0.0)
        scores = scores[valid]
        if not len(relevance):
            continue
        count = min(len(relevance), int(k) if k is not None else len(relevance))
        discounts = 1.0 / np.log2(np.arange(2, count + 2))
        ranking = np.argsort(scores)[::-1][:count]
        ideal = np.argsort(relevance)[::-1][:count]
        gain = np.sum((2.0 ** relevance[ranking] - 1.0) * discounts)
        ideal_gain = np.sum((2.0 ** relevance[ideal] - 1.0) * discounts)
        values.append(gain / ideal_gain if ideal_gain > 0.0 else 1.0)
    return float(np.mean(values)) if values else float("nan")

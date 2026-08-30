from __future__ import annotations

import math

import numpy as np


def json_ready(value):
    """Convert NumPy values and non-finite floats to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(labels > 0.5))
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(np.sum(ranks[labels > 0.5]))
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        upper_inclusive = index == bins - 1
        mask = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1] if upper_inclusive else probabilities < edges[index + 1]
        )
        if np.any(mask):
            error += float(np.mean(mask)) * abs(float(np.mean(labels[mask])) - float(np.mean(probabilities[mask])))
    return error


def fit_temperature(labels: np.ndarray, logits: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    if not len(labels) or np.all(labels == labels[0]):
        return 1.0
    temperatures = np.exp(np.linspace(math.log(0.2), math.log(5.0), 240))
    losses = []
    for temperature in temperatures:
        scaled = np.clip(logits / temperature, -30.0, 30.0)
        losses.append(float(np.mean(np.logaddexp(0.0, scaled) - labels * scaled)))
    return float(temperatures[int(np.argmin(losses))])


def fit_ensemble_temperature(labels: np.ndarray, member_logits: np.ndarray) -> float:
    """Calibrate the same mean-of-member-probabilities used by online inference."""
    labels = np.asarray(labels, dtype=np.float64)
    member_logits = np.asarray(member_logits, dtype=np.float64)
    if member_logits.ndim != 2 or member_logits.shape[1] != len(labels):
        raise ValueError("member_logits must have shape ensemble x examples")
    if not len(labels) or np.all(labels == labels[0]):
        return 1.0
    temperatures = np.exp(np.linspace(math.log(0.2), math.log(5.0), 240))
    losses = []
    for temperature in temperatures:
        probabilities = np.mean(
            1.0 / (1.0 + np.exp(-np.clip(member_logits / temperature, -30.0, 30.0))), axis=0
        )
        probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
        losses.append(
            float(-np.mean(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities)))
        )
    return float(temperatures[int(np.argmin(losses))])


def risk_report(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.float64) > 0.5
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = probabilities >= threshold
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    tn = int(np.sum(~predicted & ~labels))
    return {
        "examples": len(labels),
        "positives": int(np.sum(labels)),
        "auroc": binary_auroc(labels, probabilities),
        "brier": float(np.mean((probabilities - labels.astype(np.float64)) ** 2)),
        "ece": expected_calibration_error(labels, probabilities),
        "accuracy": float(np.mean(predicted == labels)),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "threshold": threshold,
    }

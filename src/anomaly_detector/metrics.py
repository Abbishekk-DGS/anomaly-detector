"""Evaluation the way anomaly-detection papers do it, plus event-level recall."""

from __future__ import annotations

import numpy as np


def segments(labels: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of 1s as (start, end_exclusive)."""
    padded = np.concatenate([[0], labels.astype(int), [0]])
    diff = np.diff(padded)
    return list(zip(np.where(diff == 1)[0], np.where(diff == -1)[0], strict=True))


def point_adjust(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """If any point inside a true anomaly segment is flagged, the whole segment counts as
    detected (Xu et al., 2018). Standard, but generous, so we also report raw F1."""
    adjusted = pred.copy()
    for s, e in segments(labels):
        if pred[s:e].any():
            adjusted[s:e] = 1
    return adjusted


def prf(pred: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    tp = int((pred & labels).sum())
    fp = int((pred & (1 - labels)).sum())
    fn = int(((1 - pred) & labels).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    labels = labels.astype(int)
    p, r, f = prf(pred, labels)
    pa_p, pa_r, pa_f = prf(point_adjust(pred, labels), labels)
    segs = segments(labels)
    detected = sum(bool(pred[s:e].any()) for s, e in segs)
    # false alarms counted as alert *episodes* outside true anomalies, since that's what wakes people up
    false_episodes = sum(1 for s, e in segments(pred) if not labels[s:e].any())
    return {
        "threshold": float(threshold),
        "precision": p,
        "recall": r,
        "f1": f,
        "pa_f1": pa_f,
        "events_detected": f"{detected}/{len(segs)}",
        "false_alarm_episodes": false_episodes,
    }


def best_threshold(scores: np.ndarray, labels: np.ndarray, key: str = "pa_f1") -> dict:
    """Oracle threshold search. Reported only as an upper bound; calibrated thresholds are
    what you'd deploy (see `threshold_from_train`)."""
    candidates = np.unique(np.quantile(scores, np.linspace(0.80, 0.9995, 200)))
    return max((evaluate(scores, labels, t) for t in candidates), key=lambda m: m[key])


def threshold_from_train(train_scores: np.ndarray, false_alarm_rate: float = 0.002) -> float:
    """Deployable threshold: the score exceeded by only `false_alarm_rate` of normal data."""
    return float(np.quantile(train_scores, 1 - false_alarm_rate))

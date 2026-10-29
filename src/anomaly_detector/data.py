"""Synthetic sensor data with labelled anomalies, so every detector can be scored honestly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ANOMALY_KINDS = ("spike", "level_shift", "noise_burst", "drift", "flatline", "seasonal_break")


@dataclass
class Series:
    values: np.ndarray
    labels: np.ndarray  # 1 = anomalous point
    events: list[tuple[int, int, str]]  # (start, end_exclusive, kind)
    timestamps: pd.DatetimeIndex

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"timestamp": self.timestamps, "value": self.values, "label": self.labels}
        )


def make_series(
    n: int = 5000,
    period: int = 288,  # one day at 5-minute resolution
    n_events: int = 12,
    seed: int = 0,
    train_fraction: float = 0.3,
) -> Series:
    """Daily + weekly seasonality, slow trend and noise, with anomalies injected only after
    `train_fraction` so the start of the series can serve as clean training data."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = (
        20
        + 6 * np.sin(2 * np.pi * t / period)
        + 2 * np.sin(2 * np.pi * t / (period * 7))
        + 0.0005 * t
        + rng.normal(0, 0.6, n)
    )
    values, labels = base.copy(), np.zeros(n, dtype=int)
    events: list[tuple[int, int, str]] = []
    start_min = int(n * train_fraction)
    slots = np.sort(rng.choice(np.arange(start_min, n - 200, 120), n_events, replace=False))
    for i, s in enumerate(slots):
        kind = ANOMALY_KINDS[i % len(ANOMALY_KINDS)]
        if kind == "spike":
            e = s + 1
            values[s] += rng.choice([-1, 1]) * rng.uniform(6, 10)
        elif kind == "level_shift":
            e = s + rng.integers(40, 80)
            values[s:e] += rng.choice([-1, 1]) * rng.uniform(4, 6)
        elif kind == "noise_burst":
            e = s + rng.integers(30, 60)
            values[s:e] += rng.normal(0, 3, e - s)
        elif kind == "drift":
            e = s + rng.integers(60, 100)
            values[s:e] += np.linspace(0, rng.uniform(5, 8), e - s)
        elif kind == "flatline":  # stuck sensor
            e = s + rng.integers(40, 80)
            values[s:e] = values[s]
        else:  # seasonal_break: the daily cycle inverts
            e = s + rng.integers(50, 90)
            values[s:e] = 2 * base[s:e].mean() - base[s:e]
        labels[s:e] = 1
        events.append((int(s), int(e), kind))
    ts = pd.date_range("2026-01-01", periods=n, freq="5min")
    return Series(values, labels, events, ts)

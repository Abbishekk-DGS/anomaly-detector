"""Three detectors with one interface: fit(train) then score(series) -> one score per point.

They fail in different ways, which is why the ensemble works:
- RobustZScore: instant, great for spikes; blind to slow drift and stuck sensors.
- IsolationForestDetector: window features (level, spread, slope, seasonal residual);
  catches shape changes like flatlines and noise bursts.
- LSTMAutoencoder: learns what a normal window looks like; flags windows it can't reconstruct.
"""

from __future__ import annotations

import numpy as np


def sliding_windows(x: np.ndarray, size: int) -> np.ndarray:
    """(n, size) windows ending at each point; the first size-1 points are edge-padded."""
    padded = np.concatenate([np.full(size - 1, x[0]), x])
    return np.lib.stride_tricks.sliding_window_view(padded, size)


def seasonal_residual(x: np.ndarray, period: int, train: np.ndarray | None = None) -> np.ndarray:
    """Remove the daily profile learned from training data (median per phase)."""
    ref = train if train is not None else x
    phase_ref = np.arange(len(ref)) % period
    profile = np.array([np.median(ref[phase_ref == p]) for p in range(period)])
    return x - profile[np.arange(len(x)) % period]


class RobustZScore:
    """|x - rolling median| / (1.4826 * rolling MAD), computed on the de-seasonalized signal."""

    name = "robust_z"

    def __init__(self, window: int = 60, period: int | None = 288):
        self.window, self.period = window, period
        self.train: np.ndarray | None = None

    def fit(self, train: np.ndarray) -> RobustZScore:
        self.train = train
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        r = seasonal_residual(x, self.period, self.train) if self.period else x
        w = sliding_windows(r, self.window)
        med = np.median(w, axis=1)
        mad = np.median(np.abs(w - med[:, None]), axis=1)
        floor = 1.4826 * np.median(mad) + 1e-9  # avoid blow-ups on flat windows
        return np.abs(r - med) / np.maximum(1.4826 * mad, floor)


def window_features(x: np.ndarray, size: int, period: int | None, train=None) -> np.ndarray:
    r = seasonal_residual(x, period, train) if period else x - np.median(x)
    w = sliding_windows(x, size)
    wr = sliding_windows(r, size)
    t = np.arange(size) - (size - 1) / 2
    slope = (w * t).sum(axis=1) / (t**2).sum()
    diffs = np.abs(np.diff(w, axis=1))
    return np.column_stack(
        [
            wr.mean(axis=1),  # level vs normal profile
            wr.std(axis=1),  # spread
            slope,  # trend
            diffs.mean(axis=1),  # roughness (noise bursts up, flatlines down)
            (diffs < 1e-6).mean(axis=1),  # fraction of exactly-repeated values
            np.abs(r),  # current point
        ]
    )


class IsolationForestDetector:
    name = "iforest"

    def __init__(self, window: int = 30, period: int | None = 288, seed: int = 0):
        self.window, self.period, self.seed = window, period, seed

    def fit(self, train: np.ndarray) -> IsolationForestDetector:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        self.train = train
        feats = window_features(train, self.window, self.period, train)
        self.scaler = StandardScaler().fit(feats)
        self.model = IsolationForest(n_estimators=300, random_state=self.seed)
        self.model.fit(self.scaler.transform(feats))
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        feats = self.scaler.transform(window_features(x, self.window, self.period, self.train))
        return -self.model.score_samples(feats)  # higher = more anomalous


class LSTMAutoencoder:
    """Seq2seq LSTM autoencoder. Trained on normal windows only; anomaly score is the
    reconstruction error of the window ending at each point."""

    name = "lstm_ae"

    def __init__(
        self,
        window: int = 48,
        hidden: int = 32,
        epochs: int = 15,
        seed: int = 0,
        device: str | None = None,
    ):
        self.window, self.hidden, self.epochs, self.seed = window, hidden, epochs, seed
        self.device = device

    def _build(self):
        import torch
        from torch import nn

        class AE(nn.Module):
            def __init__(self, hidden: int):
                super().__init__()
                self.enc = nn.LSTM(1, hidden, batch_first=True)
                self.dec = nn.LSTM(hidden, hidden, batch_first=True)
                self.out = nn.Linear(hidden, 1)

            def forward(self, x):
                _, (h, _) = self.enc(x)  # compress the whole window into one vector
                z = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
                y, _ = self.dec(z)
                return self.out(y)

        torch.manual_seed(self.seed)
        return AE(self.hidden)

    def fit(self, train: np.ndarray) -> LSTMAutoencoder:
        import torch

        self.mu, self.sigma = float(train.mean()), float(train.std() + 1e-9)
        windows = sliding_windows((train - self.mu) / self.sigma, self.window)[self.window :]
        self.device = self.device or "cpu"  # small model: CPU beats MPS launch overhead
        self.model = self._build().to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        data = torch.tensor(windows, dtype=torch.float32).unsqueeze(-1)
        g = torch.Generator().manual_seed(self.seed)
        self.model.train()
        for _ in range(self.epochs):
            for idx in torch.randperm(len(data), generator=g).split(64):
                batch = data[idx].to(self.device)
                loss = torch.mean((self.model(batch) - batch) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        import torch

        windows = sliding_windows((x - self.mu) / self.sigma, self.window)
        data = torch.tensor(windows, dtype=torch.float32).unsqueeze(-1)
        self.model.eval()
        errs = []
        with torch.no_grad():
            for batch in data.split(512):
                rec = self.model(batch.to(self.device)).cpu()
                errs.append(((rec - batch) ** 2).mean(dim=(1, 2)).numpy())
        return np.concatenate(errs)


def rank_normalize(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map scores to their percentile within the reference (training) scores."""
    ref = np.sort(reference)
    return np.searchsorted(ref, scores, side="right") / len(ref)


class Ensemble:
    """Fuse detectors on a common scale. Each score becomes its percentile against that
    detector's own scores on normal training data; the ensemble takes the max, so a point
    is anomalous if any specialist is confident."""

    name = "ensemble"

    def __init__(self, detectors: list):
        self.detectors = detectors

    def fit(self, train: np.ndarray, calibration: np.ndarray | None = None) -> Ensemble:
        """Fit on `train`; build each detector's reference distribution on held-out normal
        data (`calibration`). Scoring the training data itself would be optimistic, the
        autoencoder in particular reconstructs what it memorized unusually well."""
        ref_data = calibration if calibration is not None else train
        self.refs = []
        for d in self.detectors:
            d.fit(train)
            self.refs.append(d.score(ref_data))
        return self

    def score_all(self, x: np.ndarray) -> dict[str, np.ndarray]:
        out = {}
        for d, ref in zip(self.detectors, self.refs, strict=True):
            out[d.name] = rank_normalize(d.score(x), ref)
        out[self.name] = np.max(np.stack(list(out.values())), axis=0)
        return out

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.score_all(x)[self.name]

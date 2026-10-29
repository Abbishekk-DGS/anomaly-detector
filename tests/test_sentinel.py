import numpy as np

from anomaly_detector.data import make_series
from anomaly_detector.detectors import (
    Ensemble,
    IsolationForestDetector,
    LSTMAutoencoder,
    RobustZScore,
    rank_normalize,
    sliding_windows,
)
from anomaly_detector.metrics import evaluate, point_adjust, segments, threshold_from_train


def test_windows_align_with_points():
    w = sliding_windows(np.arange(5.0), 3)
    assert w.shape == (5, 3) and list(w[-1]) == [2, 3, 4] and list(w[0]) == [0, 0, 0]


def test_generator_labels_match_events():
    s = make_series(n=3000, n_events=6, seed=1)
    assert s.labels.sum() == sum(e - b for b, e, _ in s.events)
    assert s.labels[: int(3000 * 0.3)].sum() == 0  # clean training prefix


def test_segments_and_point_adjust():
    labels = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0])
    pred = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1])
    assert segments(labels) == [(1, 4), (6, 8)]
    assert list(point_adjust(pred, labels)) == [0, 1, 1, 1, 0, 0, 0, 0, 1]


def test_evaluate_counts_false_alarm_episodes():
    labels = np.array([0, 0, 1, 1, 0, 0, 0, 0])
    scores = np.array([0, 0, 0.9, 0.1, 0, 0.8, 0.8, 0])
    m = evaluate(scores, labels, 0.5)
    assert m["events_detected"] == "1/1" and m["false_alarm_episodes"] == 1


def test_robust_z_catches_spike():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    x[300] += 12
    scores = RobustZScore(period=None).fit(x[:200]).score(x)
    assert int(np.argmax(scores)) == 300


def test_iforest_catches_flatline():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 1200)
    x[800:860] = x[800]
    det = IsolationForestDetector(window=20, period=None).fit(x[:600])
    scores = det.score(x)
    assert scores[820:860].mean() > np.quantile(scores[:600], 0.99)


def test_rank_normalize_and_threshold():
    ref = np.arange(1000.0)
    assert rank_normalize(np.array([-1.0, 499.5, 5000.0]), ref).tolist() == [0.0, 0.5, 1.0]
    assert abs(threshold_from_train(ref, 0.01) - 989.01) < 0.1


def test_ensemble_end_to_end_beats_chance():
    s = make_series(n=2400, period=96, n_events=6, seed=3)
    split = 720
    ens = Ensemble([RobustZScore(period=96), LSTMAutoencoder(window=24, epochs=3)])
    ens.fit(s.values[:split])
    train_scores = ens.score_all(s.values[:split])["ensemble"]
    scores = ens.score(s.values[split:])
    m = evaluate(scores, s.labels[split:], threshold_from_train(train_scores, 0.01))
    hit, total = map(int, m["events_detected"].split("/"))
    assert hit >= total // 2 and m["pa_f1"] > 0.5

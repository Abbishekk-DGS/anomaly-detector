"""anomaly-detector: find anomalies in time series.

anomaly-detector bench                                 # score every detector on labelled data
anomaly-detector detect sensor.csv --column temp --train-rows 2000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import __version__
from .data import make_series
from .detectors import Ensemble, IsolationForestDetector, LSTMAutoencoder, RobustZScore
from .metrics import best_threshold, evaluate, threshold_from_train


def build(period: int | None) -> Ensemble:
    return Ensemble(
        [
            RobustZScore(period=period),
            IsolationForestDetector(period=period),
            LSTMAutoencoder(),
        ]
    )


def cmd_bench(args) -> None:
    rows = []
    for seed in range(args.seeds):
        s = make_series(seed=seed)
        split = int(len(s.values) * 0.3)
        train, test, labels = s.values[:split], s.values[split:], s.labels[split:]
        fit_end = int(split * 2 / 3)  # fit on 2/3 of the clean prefix, calibrate on the rest
        ens = build(period=288)
        t0 = time.time()
        ens.fit(train[:fit_end], calibration=train[fit_end:])
        train_scores = ens.score_all(train[fit_end:])
        test_scores = ens.score_all(test)
        elapsed = time.time() - t0
        for name, sc in test_scores.items():
            thr = threshold_from_train(train_scores[name], args.false_alarm_rate)
            m = evaluate(sc, labels, thr)
            oracle = best_threshold(sc, labels)
            rows.append({"seed": seed, "detector": name, **m, "oracle_pa_f1": oracle["pa_f1"]})
        print(f"seed {seed}: fitted + scored 3 detectors in {elapsed:.1f}s")

    df = pd.DataFrame(rows)
    df["events_hit"] = df["events_detected"].str.split("/").str[0].astype(int)
    df["events_total"] = df["events_detected"].str.split("/").str[1].astype(int)
    summary = df.groupby("detector", sort=False).agg(
        f1=("f1", "mean"),
        pa_f1=("pa_f1", "mean"),
        oracle_pa_f1=("oracle_pa_f1", "mean"),
        events_hit=("events_hit", "sum"),
        events_total=("events_total", "sum"),
        false_alarms=("false_alarm_episodes", "sum"),
    )
    summary["events"] = (
        summary["events_hit"].astype(str) + "/" + summary["events_total"].astype(str)
    )
    print(
        f"\nMean over {args.seeds} seeds (threshold calibrated on normal data, no labels used):\n"
    )
    print(summary[["f1", "pa_f1", "oracle_pa_f1", "events", "false_alarms"]].round(3).to_markdown())


def cmd_detect(args) -> None:
    df = pd.read_csv(args.csv)
    x = df[args.column].astype(float).interpolate().to_numpy()
    train_rows = args.train_rows or int(len(x) * 0.3)
    fit_end = int(train_rows * 2 / 3)
    ens = build(period=args.period or None)
    ens.fit(x[:fit_end], calibration=x[fit_end:train_rows])
    train_scores = ens.score_all(x[fit_end:train_rows])
    scores = ens.score_all(x)
    for name, sc in scores.items():
        df[f"score_{name}"] = np.round(sc, 4)
    thr = threshold_from_train(train_scores["ensemble"], args.false_alarm_rate)
    df["anomaly"] = (scores["ensemble"] >= thr).astype(int)
    df.to_csv(args.out, index=False)
    print(f"{int(df['anomaly'].sum())} anomalous points (threshold {thr:.3f}) -> {args.out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="anomaly-detector",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__} · by Abbishekk DGS · abbishekkdgs.tech",
    )
    parser.add_argument("--false-alarm-rate", type=float, default=0.002)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--seeds", type=int, default=3)
    b.set_defaults(fn=cmd_bench)
    d = sub.add_parser("detect")
    d.add_argument("csv")
    d.add_argument("--column", default="value")
    d.add_argument("--train-rows", type=int, default=None, help="leading rows known to be normal")
    d.add_argument("--period", type=int, default=0, help="seasonal period in rows (0 = none)")
    d.add_argument("--out", default="anomalies.csv")
    d.set_defaults(fn=cmd_detect)
    args = parser.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Step 9: train the final models on ALL data and save them for the live pipeline.

Chosen setup: additive residual on the best naive baseline, no month feature,
regularized LightGBM. Horizons 6h, 12h and 24h (the 1h forecast just uses
persistence, because the model did not beat it).

Keep this file in the same folder as baseline.py and experiments.py.
Run:
    venv\\Scripts\\python train_final.py
"""
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from baseline import DATA, build_features
from experiments import REGULARIZED

HORIZONS = [6, 12, 24]
ROUNDS = 200
MODELS = Path("models")


def naive_lag(h):
    """Best simple baseline: persistence up to 6h, same hour yesterday for 12h+."""
    return 0 if h <= 6 else 24 - h


if __name__ == "__main__":
    df = pd.read_csv(DATA, parse_dates=["time"], index_col="time")
    X, g = build_features(df)
    Xm = X.drop(columns="month")
    MODELS.mkdir(exist_ok=True)

    card = {
        "description": "PM2.5 forecast for Delhi (Chandni Chowk). Prediction = naive "
                       "baseline + learned correction, clipped at 0.",
        "features": list(Xm.columns),
        "trained_through": str(df.index.max()),
        "lightgbm_params": REGULARIZED,
        "boosting_rounds": ROUNDS,
        "leave_one_winter_out_avg_MAE_improvement_pct": {"1": "no gain (use persistence)",
                                                          "6": 13.5, "12": 9.7, "24": 6.2},
        "known_weakness": "Under-predicts in winters more severe than any seen in training "
                          "(e.g. 2025-26 when held out). Gains are smaller and less reliable "
                          "at 12h and 24h than at 6h.",
        "horizons": {},
    }

    for h in HORIZONS:
        y = g.shift(-h)
        naive = g.shift(naive_lag(h))
        valid = y.notna() & X["pm25_lag0"].notna() & naive.notna()
        idx = valid[valid].index

        booster = lgb.train(
            REGULARIZED,
            lgb.Dataset(Xm.loc[idx], label=y.loc[idx] - naive.loc[idx]),
            num_boost_round=ROUNDS,
        )
        path = MODELS / f"pm25_h{h}.txt"
        booster.save_model(str(path))
        card["horizons"][str(h)] = {
            "naive_lag": naive_lag(h),
            "train_rows": int(len(idx)),
            "model_file": path.name,
        }
        print(f"h={h:>2}: trained on {len(idx)} rows, saved to {path}")

    with open(MODELS / "model_card.json", "w") as f:
        json.dump(card, f, indent=2)
    print(f"Saved {MODELS / 'model_card.json'}")

    # Self-check: reload from disk and forecast from the most recent usable row
    last = Xm.dropna(subset=["pm25_lag0"]).index[-1]
    print(f"\nSelf-check using the data at {last} (reloaded from disk):")
    print(f"current PM2.5: {Xm.loc[last, 'pm25_lag0']:.1f}")
    for h in HORIZONS:
        b = lgb.Booster(model_file=str(MODELS / f"pm25_h{h}.txt"))
        base = g.shift(naive_lag(h)).loc[last]
        pred = max(0.0, base + b.predict(Xm.loc[[last]])[0])
        print(f"  +{h:>2}h: baseline {base:.1f} -> model forecast {pred:.1f}")
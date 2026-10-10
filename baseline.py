"""Step 5: build features, run simple baselines, train a first LightGBM model.

Install once:
    venv\\Scripts\\python -m pip install lightgbm
Run:
    venv\\Scripts\\python baseline.py
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

DATA = Path("data/processed/delhi_hourly.csv")
HORIZONS = [1, 6, 12, 24]                  # hours ahead to forecast
TEST_START = pd.Timestamp("2025-10-01")    # everything from here on is the test set
WINTER = [11, 12, 1, 2]
# CPCB-style PM2.5 category edges: Good, Satisfactory, Moderate, Poor, Very Poor, Severe
BINS = [-np.inf, 30, 60, 90, 120, 250, np.inf]


def build_features(df):
    """All features use only information available at time t (no peeking ahead)."""
    g = df.groupby("segment")["pm25"]          # shifts never cross the 2022-2025 hole
    X = pd.DataFrame(index=df.index)

    for k in [0, 1, 2, 3, 6, 12, 24]:
        X[f"pm25_lag{k}"] = g.shift(k)
    X["pm25_roll6"] = g.transform(lambda s: s.rolling(6, min_periods=4).mean())
    X["pm25_roll24"] = g.transform(lambda s: s.rolling(24, min_periods=18).mean())
    X["pm25_change3"] = X["pm25_lag0"] - X["pm25_lag3"]

    for col in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                "precipitation", "surface_pressure"]:
        X[col] = df[col]
    rad = np.deg2rad(df["wind_direction_10m"])
    X["wind_dir_sin"], X["wind_dir_cos"] = np.sin(rad), np.cos(rad)

    X["hour"] = df.index.hour
    X["month"] = df.index.month
    X["dayofweek"] = df.index.dayofweek
    return X, g


def evaluate(y_true, y_pred):
    err = y_pred - y_true
    mae = np.abs(err).mean()
    rmse = np.sqrt((err ** 2).mean())
    cat = (pd.cut(y_true, BINS, labels=False) == pd.cut(y_pred, BINS, labels=False)).mean()
    return round(mae, 1), round(rmse, 1), round(cat * 100, 1)


if __name__ == "__main__":
    df = pd.read_csv(DATA, parse_dates=["time"], index_col="time")
    X, g = build_features(df)

    rows, importances = [], None
    for h in HORIZONS:
        y = g.shift(-h)                                    # PM2.5 h hours from now
        valid = y.notna() & X["pm25_lag0"].notna()
        Xh, yh = X[valid], y[valid]

        is_train = Xh.index < TEST_START
        Xtr, ytr, Xte, yte = Xh[is_train], yh[is_train], Xh[~is_train], yh[~is_train]
        print(f"h={h:>2}: train rows {len(Xtr)} "
              f"(winter {Xtr['month'].isin(WINTER).sum()}), "
              f"test rows {len(Xte)} (winter {Xte['month'].isin(WINTER).sum()})")

        # Baseline 1: tomorrow looks like right now.
        persist = Xte["pm25_lag0"]
        # Baseline 2: same hour yesterday (value at target time minus 24h).
        yesterday = g.shift(24 - h).loc[Xte.index]

        # Model: trained on log(1+PM2.5) so huge spikes don't dominate training.
        # Native LightGBM API: works without scikit-learn.
        params = {
            "objective": "regression", "learning_rate": 0.05, "num_leaves": 31,
            "min_data_in_leaf": 40, "bagging_fraction": 0.8, "bagging_freq": 1,
            "feature_fraction": 0.8, "seed": 42, "verbose": -1,
        }
        booster = lgb.train(params, lgb.Dataset(Xtr, label=np.log1p(ytr)),
                            num_boost_round=400)
        pred = pd.Series(np.expm1(booster.predict(Xte)), index=Xte.index)
        if h == HORIZONS[-1]:
            importances = pd.Series(booster.feature_importance(), index=Xte.columns)

        for subset, mask in [("all test", Xte.index == Xte.index),
                             ("winter only", Xte["month"].isin(WINTER).values)]:
            ok = mask & yesterday.notna().values       # compare on identical rows
            if ok.sum() == 0:
                continue
            for name, p in [("persistence", persist), ("same hour yesterday", yesterday),
                            ("LightGBM", pred)]:
                mae, rmse, cat = evaluate(yte[ok], p[ok])
                rows.append({"horizon_h": h, "subset": subset, "method": name,
                             "n": int(ok.sum()), "MAE": mae, "RMSE": rmse,
                             "category_acc_%": cat})

    print("\n" + pd.DataFrame(rows).to_string(index=False))
    print("\nTop features for the 24h model:")
    print(importances.sort_values(ascending=False).head(8).to_string())
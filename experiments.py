"""Step 6: find out why LightGBM loses to simple baselines at 12-24h, and test fixes.

Keep this file in the same folder as baseline.py (it imports from it).
Run:
    venv\\Scripts\\python experiments.py
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from baseline import DATA, TEST_START, WINTER, build_features, evaluate

HORIZONS = [1, 6, 12, 24]
WEATHER = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
           "precipitation", "surface_pressure"]

DEFAULT = {
    "objective": "regression", "learning_rate": 0.05, "num_leaves": 31,
    "min_data_in_leaf": 40, "bagging_fraction": 0.8, "bagging_freq": 1,
    "feature_fraction": 0.8, "seed": 42, "verbose": -1,
}
REGULARIZED = {**DEFAULT, "num_leaves": 15, "min_data_in_leaf": 100}


def fit_predict(Xtr, ytr, Xte, naive_tr, naive_te, mode, params, rounds):
    if mode == "log":                       # predict log(1+PM2.5)
        m = lgb.train(params, lgb.Dataset(Xtr, label=np.log1p(ytr)), rounds)
        return np.expm1(m.predict(Xte))
    if mode == "raw":                       # predict PM2.5 directly
        m = lgb.train(params, lgb.Dataset(Xtr, label=ytr), rounds)
        return m.predict(Xte)
    # "resid": predict the correction to the best simple baseline
    m = lgb.train(params, lgb.Dataset(Xtr, label=ytr - naive_tr), rounds)
    return np.clip(naive_te + m.predict(Xte), 0, None)


if __name__ == "__main__":
    df = pd.read_csv(DATA, parse_dates=["time"], index_col="time")
    X, g = build_features(df)
    gw = df.groupby("segment")

    rows = []
    for h in HORIZONS:
        y = g.shift(-h)
        # Best simple baseline: persistence for short horizons, same hour yesterday for 12h+
        naive = g.shift(0 if h <= 6 else 24 - h)

        # Same features plus the weather at the target hour (reanalysis stands in
        # for a forecast, so treat this variant as an upper bound)
        Xf = X.copy()
        for c in WEATHER:
            Xf[f"{c}_fut"] = gw[c].shift(-h)

        valid = y.notna() & X["pm25_lag0"].notna() & naive.notna()
        idx = valid[valid].index
        tr_idx, te_idx = idx[idx < TEST_START], idx[idx >= TEST_START]
        winter = X.loc[te_idx, "month"].isin(WINTER).values
        y_te = y.loc[te_idx]

        variants = [
            ("V0 current (log target)", X, "log", DEFAULT, 400),
            ("V1 raw target", X, "raw", DEFAULT, 400),
            ("V2 residual vs baseline", X, "resid", DEFAULT, 400),
            ("V3 residual, no month", X.drop(columns="month"), "resid", DEFAULT, 400),
            ("V4 V3 + regularized", X.drop(columns="month"), "resid", REGULARIZED, 200),
            ("V5 V4 + future weather", Xf.drop(columns="month"), "resid", REGULARIZED, 200),
        ]

        preds = {"best naive baseline": naive.loc[te_idx]}
        for name, frame, mode, params, rounds in variants:
            p = fit_predict(frame.loc[tr_idx], y.loc[tr_idx], frame.loc[te_idx],
                            naive.loc[tr_idx], naive.loc[te_idx], mode, params, rounds)
            preds[name] = pd.Series(p, index=te_idx)

        for name, p in preds.items():
            mae_all, _, _ = evaluate(y_te, p)
            mae_w, rmse_w, cat_w = evaluate(y_te[winter], p[winter])
            rows.append({
                "horizon_h": h, "variant": name,
                "MAE_all": mae_all, "MAE_winter": mae_w, "RMSE_winter": rmse_w,
                "bias_winter": round((p[winter] - y_te[winter]).mean(), 1),
                "cat_acc_winter_%": cat_w,
            })

    print(pd.DataFrame(rows).to_string(index=False))
    print("\nbias_winter < 0 means the model under-predicts on average in winter.")
    print("V5 uses real weather at the target hour, so it is an upper bound, "
          "not a fair estimate of a live forecast.")
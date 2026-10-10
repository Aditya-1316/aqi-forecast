"""Step 8: does predicting a *ratio* to the baseline fix under-prediction in severe winters?

Compares two ways of correcting the best naive baseline, using leave-one-winter-out:
  - additive residual:  prediction = baseline + correction (what we had)
  - log-ratio residual: prediction = baseline * exp(correction), roughly. The correction
    scales with the pollution level, so it can extrapolate to worse winters.

Keep this file in the same folder as baseline.py, experiments.py and diagnose.py.
Run:
    venv\\Scripts\\python compare_targets.py
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from baseline import DATA, build_features, evaluate
from diagnose import MIN_TEST_ROWS, season_labels
from experiments import REGULARIZED, fit_predict

HORIZONS = [1, 6, 12, 24]


def fit_predict_logratio(Xtr, ytr, Xte, naive_tr, naive_te, params, rounds):
    target = np.log1p(ytr) - np.log1p(naive_tr)
    m = lgb.train(params, lgb.Dataset(Xtr, label=target), rounds)
    return np.expm1(np.log1p(naive_te) + m.predict(Xte))


if __name__ == "__main__":
    df = pd.read_csv(DATA, parse_dates=["time"], index_col="time")
    X, g = build_features(df)
    labels = season_labels(df.index, df["segment"])
    Xm = X.drop(columns="month")

    rows = []
    for h in HORIZONS:
        y = g.shift(-h)
        naive = g.shift(0 if h <= 6 else 24 - h)
        valid = y.notna() & X["pm25_lag0"].notna() & naive.notna()
        idx = valid[valid].index
        lab = labels.loc[idx]

        for w in sorted(l for l in lab.unique() if l.startswith("winter")):
            in_test = (lab == w).values
            if in_test.sum() < MIN_TEST_ROWS:
                continue
            te_idx = idx[in_test]
            t0, t1 = te_idx.min(), te_idx.max()
            near = (idx >= t0 - pd.Timedelta(days=2)) & (idx <= t1 + pd.Timedelta(days=2))
            tr_idx = idx[~near]

            args = (Xm.loc[tr_idx], y.loc[tr_idx], Xm.loc[te_idx],
                    naive.loc[tr_idx], naive.loc[te_idx])
            preds = {
                "additive residual": pd.Series(
                    fit_predict(*args, "resid", REGULARIZED, 200), index=te_idx),
                "log-ratio residual": pd.Series(
                    fit_predict_logratio(*args, REGULARIZED, 200), index=te_idx),
            }

            y_te = y.loc[te_idx]
            mae_n, _, cat_n = evaluate(y_te, naive.loc[te_idx])
            for method, pred in preds.items():
                mae_m, _, cat_m = evaluate(y_te, pred)
                rows.append({
                    "horizon_h": h, "test_winter": w.replace("winter ", ""),
                    "method": method, "MAE_naive": mae_n, "MAE_model": mae_m,
                    "bias": round((pred - y_te).mean(), 1),
                    "cat_acc_naive_%": cat_n, "cat_acc_model_%": cat_m,
                    "MAE_improvement_%": round((1 - mae_m / mae_n) * 100, 1),
                })

    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    print("\nAverage MAE improvement over the naive baseline, across winters:")
    print(res.pivot_table(index="horizon_h", columns="method",
                          values="MAE_improvement_%", aggfunc="mean").round(1).to_string())
    print("\nOnly the severe 2025-2026 winter (the one that broke the earlier model):")
    sev = res[res["test_winter"] == "2025-2026"]
    print(sev.pivot_table(index="horizon_h", columns="method",
                          values=["MAE_improvement_%", "bias"]).round(1).to_string())
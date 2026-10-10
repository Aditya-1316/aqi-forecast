"""Step 7: check whether the winter-2025-26 test result is a fluke or a real pattern.

Part 1: how different are the winters from each other (and the feeds outside winter)?
Part 2: leave-one-winter-out: train on everything except one winter, test on that winter.

Keep this file in the same folder as baseline.py and experiments.py.
Run:
    venv\\Scripts\\python diagnose.py
"""
import pandas as pd

from baseline import DATA, build_features, evaluate
from experiments import REGULARIZED, fit_predict

HORIZONS = [1, 6, 12, 24]
MIN_TEST_ROWS = 200


def season_labels(index, segment):
    labels = []
    for ts, seg in zip(index, segment):
        if ts.month in (11, 12):
            labels.append(f"winter {ts.year}-{ts.year + 1}")
        elif ts.month in (1, 2):
            labels.append(f"winter {ts.year - 1}-{ts.year}")
        else:
            labels.append(f"non-winter, feed {seg}")
    return pd.Series(labels, index=index)


if __name__ == "__main__":
    df = pd.read_csv(DATA, parse_dates=["time"], index_col="time")
    X, g = build_features(df)
    labels = season_labels(df.index, df["segment"])

    # ---- Part 1: distribution of PM2.5 per season ----
    stats = df.assign(label=labels).groupby("label")["pm25"].agg(
        hours="count",
        mean="mean",
        median="median",
        p90=lambda s: s.quantile(0.9),
        max="max",
        pct_over_250=lambda s: (s > 250).sum() / max(s.notna().sum(), 1) * 100,
    ).round(1)
    print("PM2.5 by season (hours with a reading, then ug/m3):")
    print(stats.to_string())

    # ---- Part 2: leave-one-winter-out ----
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
            # train on everything outside the test winter (plus a 2-day margin)
            near = (idx >= t0 - pd.Timedelta(days=2)) & (idx <= t1 + pd.Timedelta(days=2))
            tr_idx = idx[~near]

            p = fit_predict(Xm.loc[tr_idx], y.loc[tr_idx], Xm.loc[te_idx],
                            naive.loc[tr_idx], naive.loc[te_idx],
                            "resid", REGULARIZED, 200)
            pred = pd.Series(p, index=te_idx)
            y_te = y.loc[te_idx]

            mae_n, _, cat_n = evaluate(y_te, naive.loc[te_idx])
            mae_m, _, cat_m = evaluate(y_te, pred)
            rows.append({
                "horizon_h": h, "test_winter": w.replace("winter ", ""),
                "n": len(te_idx), "MAE_naive": mae_n, "MAE_model": mae_m,
                "bias_model": round((pred - y_te).mean(), 1),
                "cat_acc_naive_%": cat_n, "cat_acc_model_%": cat_m,
            })

    res = pd.DataFrame(rows)
    res["MAE_improvement_%"] = ((1 - res["MAE_model"] / res["MAE_naive"]) * 100).round(1)
    print("\nLeave-one-winter-out (model = residual, no month, regularized):")
    print(res.to_string(index=False))
    print("\nAverage MAE improvement over the naive baseline, across winters "
          "(positive = model is better):")
    print(res.groupby("horizon_h")["MAE_improvement_%"].mean().round(1).to_string())
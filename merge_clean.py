"""Step 4: clean the PM2.5 data, merge it with weather, and report where the gaps are.

Run:
    venv\\Scripts\\python merge_clean.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")

A_FILE = RAW / "pm25_chandni_chowk_a_39410.csv"       # old feed, ends 2022-08
B_FILE = RAW / "pm25_chandni_chowk_b_12236361.csv"    # new feed, starts 2025-02
W_FILE = RAW / "weather_delhi.csv"

MAX_VALID = 1000   # above this is treated as a sensor error
STUCK_RUN = 8      # same value for this many hours in a row = stuck sensor
FILL_LIMIT = 3     # only interpolate gaps up to this many hours


def load_pm25(path):
    df = pd.read_csv(path, parse_dates=["time"])
    df["time"] = df["time"].dt.floor("h")          # align :30 timestamps to the hour
    return df.groupby("time")["pm25"].mean()       # average any duplicates


def clean(s, label):
    """Reindex to a full hourly range, then blank out bad readings."""
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
    present = int(s.notna().sum())

    bad = (s <= 0) | (s > MAX_VALID)
    n_bad = int(bad.sum())
    s[bad] = np.nan

    run_id = (s != s.shift()).cumsum()
    run_len = s.groupby(run_id).transform("size")
    stuck = s.notna() & (run_len >= STUCK_RUN)
    n_stuck = int(stuck.sum())
    s[stuck] = np.nan

    print(f"[{label}] hours with a reading: {present}")
    print(f"[{label}] removed as <=0 or >{MAX_VALID}: {n_bad}")
    print(f"[{label}] removed as stuck (>= {STUCK_RUN} identical hours): {n_stuck}")
    return s


def gap_report(s, label, top=5):
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    runs = [(g.index[0], len(g)) for _, g in s[isna].groupby(run_id[isna])]
    runs.sort(key=lambda x: -x[1])
    print(f"[{label}] {len(runs)} gaps remain after filling short ones; "
          f"{sum(1 for _, n in runs if n > 24)} are longer than 24 hours")
    print(f"[{label}] longest gaps (start, hours):")
    for start, n in runs[:top]:
        print(f"      {start}  ->  {n} h (~{n / 24:.1f} days)")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)

    a = clean(load_pm25(A_FILE), "A")
    print()
    b = clean(load_pm25(B_FILE), "B")
    print()

    df = pd.concat(
        [
            pd.DataFrame({"pm25_clean": a, "segment": "A"}),
            pd.DataFrame({"pm25_clean": b, "segment": "B"}),
        ]
    )
    df.index.name = "time"

    # Fill only short gaps, never across the 2022-2025 hole (done per segment)
    df["pm25"] = df.groupby("segment")["pm25_clean"].transform(
        lambda x: x.interpolate(limit=FILL_LIMIT, limit_area="inside")
    )

    weather = pd.read_csv(W_FILE, parse_dates=["time"], index_col="time")
    df = df.join(weather, how="left")

    for seg in ["A", "B"]:
        part = df[df["segment"] == seg]
        print(f"=== Segment {seg}: {part.index.min()} -> {part.index.max()} ===")
        print(f"PM2.5 missing after cleaning and short fills: "
              f"{part['pm25'].isna().mean() * 100:.1f}%")
        print(f"Weather rows missing: {part['temperature_2m'].isna().sum()}")
        gap_report(part["pm25"], seg)
        print()

    monthly = df.groupby(df.index.to_period("M"))["pm25"].apply(
        lambda x: round(x.isna().mean() * 100)
    )
    print("Percent of hours missing, by month:")
    print(monthly.to_string())

    path = OUT / "delhi_hourly.csv"
    df.to_csv(path)
    print(f"\nSaved {len(df)} rows to {path}")
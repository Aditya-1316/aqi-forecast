"""Step 10: forecast pipeline (live mode and replay mode).

Live mode:   fetches the last ~3 days of PM2.5 (OpenAQ) and weather (Open-Meteo), applies
             the same cleaning and features as training, writes output/forecast.json.
Replay mode: does the same from your saved historical dataset, as if it were a past
             moment. Useful when the live sensor is offline, and for building the website.
             Replays are in-sample (the models saw that period in training), so they test
             the plumbing, not the accuracy.

Keep this file in the same folder as baseline.py and the models/ folder.
Live (PowerShell, same terminal):
    $env:OPENAQ_API_KEY="paste_your_key_here"
    venv\\Scripts\\python pipeline.py
Replay:
    venv\\Scripts\\python pipeline.py --replay "2025-12-05 08:00"
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

from baseline import build_features

SENSOR_ID = 12236361                      # Chandni Chowk PM2.5, the live feed
LOCATION_NAME = "Delhi (Chandni Chowk)"
LAT, LON = 28.6139, 77.2090
OPENAQ = "https://api.openaq.org/v3"
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                "wind_direction_10m", "precipitation", "surface_pressure"]

MODELS = Path(".")
OUT = Path("output")
REPLAY_DATA = Path("data/processed/delhi_hourly.csv")
HISTORY_HOURS = 72          # enough for the 24h lags and rolling windows
MAX_STALE_HOURS = 3         # refuse to forecast if the latest reading is older than this
MAX_VALID, STUCK_RUN, FILL_LIMIT = 1000, 8, 3   # same cleaning rules as training

CATEGORIES = [(30, "Good"), (60, "Satisfactory"), (90, "Moderate"),
              (120, "Poor"), (250, "Very Poor"), (np.inf, "Severe")]


def category(v):
    for limit, name in CATEGORIES:
        if v <= limit:
            return name


def fetch_recent_pm25():
    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        sys.exit("OPENAQ_API_KEY is not set in this terminal.")
    now = datetime.now(timezone.utc)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    params = {
        "datetime_from": (now - timedelta(hours=HISTORY_HOURS)).strftime(fmt),
        "datetime_to": now.strftime(fmt),
        "limit": 1000,
    }
    resp = requests.get(f"{OPENAQ}/sensors/{SENSOR_ID}/hours",
                        headers={"X-API-Key": key}, params=params, timeout=60)
    resp.raise_for_status()
    rows = [{"time_utc": r["period"]["datetimeFrom"]["utc"], "pm25": r["value"]}
            for r in resp.json().get("results", [])]
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    t = (pd.to_datetime(df["time_utc"], utc=True)
         .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.floor("h"))
    return pd.Series(df["pm25"].values, index=t).groupby(level=0).mean().sort_index()


def fetch_recent_weather():
    params = {
        "latitude": LAT, "longitude": LON,
        "hourly": ",".join(WEATHER_VARS),
        "past_days": 3, "forecast_days": 1,
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=60)
    resp.raise_for_status()
    w = pd.DataFrame(resp.json()["hourly"])
    w["time"] = pd.to_datetime(w["time"])
    return w.set_index("time")


def clean_series(s):
    """Same rules as training: drop bad readings, drop stuck runs, fill short gaps."""
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
    s[(s <= 0) | (s > MAX_VALID)] = np.nan
    run_id = (s != s.shift()).cumsum()
    run_len = s.groupby(run_id).transform("size")
    s[s.notna() & (run_len >= STUCK_RUN)] = np.nan
    return s.interpolate(limit=FILL_LIMIT, limit_area="inside")


def num(v):
    return None if v is None or pd.isna(v) else round(float(v), 1)


def make_forecast(raw=None, weather=None, now_local=None):
    """raw / weather / now_local are only passed in replay mode; live mode fetches them."""
    card = json.load(open(MODELS / "model_card.json"))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if raw is None:
        raw = fetch_recent_pm25()
    if raw.empty:
        return {"status": "no_data", "generated_at": generated,
                "message": "No recent PM2.5 readings were returned."}

    pm = clean_series(raw)
    valid = pm.dropna()
    if valid.empty:
        return {"status": "no_data", "generated_at": generated,
                "message": "All recent readings failed the quality checks."}

    T = valid.index[-1]                                   # latest usable hour (India time)
    if now_local is None:
        now_local = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    age_h = (now_local - T).total_seconds() / 3600
    if age_h > MAX_STALE_HOURS:
        return {"status": "stale", "generated_at": generated, "as_of": str(T),
                "message": f"Latest usable reading is {age_h:.0f} hours old, "
                           "too old for a reliable forecast."}
    pm = pm.loc[:T]

    if weather is None:
        weather = fetch_recent_weather()
    df = pd.DataFrame({"pm25": pm})
    df["segment"] = "live"
    df = df.join(weather, how="left")

    X, g = build_features(df)
    row = X.drop(columns="month")[card["features"]].loc[[T]]

    current = float(pm.loc[T])
    forecasts = [{
        "horizon_h": 1, "time": str(T + pd.Timedelta(hours=1)), "pm25": num(current),
        "category": category(current), "method": "persistence",
    }]
    for h_str, meta in sorted(card["horizons"].items(), key=lambda kv: int(kv[0])):
        h = int(h_str)
        base = g.shift(meta["naive_lag"]).loc[T]
        if pd.isna(base):
            continue                                     # not enough history for this horizon
        booster = lgb.Booster(model_file=str(MODELS / meta["model_file"]))
        pred = max(0.0, float(base + booster.predict(row)[0]))
        forecasts.append({
            "horizon_h": h, "time": str(T + pd.Timedelta(hours=h)), "pm25": num(pred),
            "category": category(pred), "method": "model",
        })

    history = [{"time": str(t), "pm25": num(v)}
               for t, v in pm.loc[T - pd.Timedelta(hours=23):T].items()]
    return {
        "status": "ok",
        "generated_at": generated,
        "location": LOCATION_NAME,
        "as_of": str(T),
        "current": {"pm25": num(current), "category": category(current)},
        "forecasts": forecasts,
        "history_24h": history,
        "model": {"trained_through": card["trained_through"],
                  "known_weakness": card["known_weakness"]},
    }


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)

    if "--replay" in sys.argv:
        as_of = pd.Timestamp(sys.argv[sys.argv.index("--replay") + 1])
        hist = pd.read_csv(REPLAY_DATA, parse_dates=["time"], index_col="time")
        part = hist.loc[as_of - pd.Timedelta(hours=HISTORY_HOURS): as_of]
        result = make_forecast(raw=part["pm25_clean"].dropna(),
                               weather=part[WEATHER_VARS], now_local=as_of)
        result["replay"] = True
        for f in result.get("forecasts", []):
            f["actual_pm25"] = num(hist["pm25_clean"].get(pd.Timestamp(f["time"])))
        out_path = OUT / "forecast_replay.json"
    else:
        result = make_forecast()
        out_path = OUT / "forecast.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "history_24h"}, indent=2))
    print(f"\nSaved {out_path}")

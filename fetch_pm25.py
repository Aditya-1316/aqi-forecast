"""Step 3: download hourly PM2.5 for the Chandni Chowk sensors from OpenAQ v3.

Setup (PowerShell, same terminal):
    $env:OPENAQ_API_KEY="paste_your_key_here"
Run:
    venv\\Scripts\\python fetch_pm25.py
"""
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.openaq.org/v3"
SENSORS = {
    "39410": "chandni_chowk_a",
    "12236361": "chandni_chowk_b",
}
START = "2020-11-01T00:00:00Z"
PAGE_LIMIT = 1000


def headers():
    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        sys.exit("OPENAQ_API_KEY is not set in this terminal.")
    return {"X-API-Key": key}


def get_json(url, params):
    """GET with simple retry/backoff in case we hit the rate limit."""
    for attempt in range(5):
        resp = requests.get(url, headers=headers(), params=params, timeout=60)
        if resp.status_code == 429:
            wait = 15 * (attempt + 1)
            print(f"  rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    sys.exit("Still rate limited after several retries. Try again in a few minutes.")


def fetch_sensor_hours(sensor_id):
    end = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, page = [], 1
    while True:
        params = {
            "datetime_from": START,
            "datetime_to": end,
            "limit": PAGE_LIMIT,
            "page": page,
        }
        data = get_json(f"{BASE}/sensors/{sensor_id}/hours", params)
        results = data.get("results", [])
        for r in results:
            rows.append(
                {
                    "time_utc": r["period"]["datetimeFrom"]["utc"],
                    "pm25": r["value"],
                }
            )
        print(f"  page {page}: {len(results)} rows")
        if len(results) < PAGE_LIMIT:
            break
        page += 1
        time.sleep(1)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Convert to India time so it lines up with the weather table
    t = pd.to_datetime(df["time_utc"], utc=True)
    df["time"] = t.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return df.drop(columns="time_utc").set_index("time").sort_index()


if __name__ == "__main__":
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)

    for sensor_id, name in SENSORS.items():
        print(f"Fetching sensor {sensor_id} ({name})...")
        df = fetch_sensor_hours(sensor_id)
        if df.empty:
            print("  no data returned\n")
            continue

        path = out / f"pm25_{name}_{sensor_id}.csv"
        df.to_csv(path)

        full_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
        missing = len(full_range) - len(df)
        print(f"  saved {len(df)} rows to {path}")
        print(f"  first: {df.index.min()}  last: {df.index.max()}")
        print(f"  missing hours inside that range: {missing} "
              f"({100 * missing / len(full_range):.1f}%)")
        print(f"  PM2.5 min/median/max: {df['pm25'].min():.1f} / "
              f"{df['pm25'].median():.1f} / {df['pm25'].max():.1f}\n")
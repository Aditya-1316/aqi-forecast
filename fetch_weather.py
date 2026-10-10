"""Step 1: download hourly historical weather for a city from Open-Meteo (no API key needed).

Usage:
    python fetch_weather.py delhi
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

# Add more cities here later; nothing else in the script is city-specific.
CITIES = {
    "delhi": (28.6139, 77.2090),
}

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "surface_pressure",
]

URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather(city: str, start: str, end: str) -> pd.DataFrame:
    lat, lon = CITIES[city]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get(URL, params=params, timeout=120)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


if __name__ == "__main__":
    city = sys.argv[1].lower() if len(sys.argv) > 1 else "delhi"
    if city not in CITIES:
        sys.exit(f"Unknown city '{city}'. Add it to CITIES first.")

    # The archive lags real time by a few days, so stop a week back.
    end = (date.today() - timedelta(days=7)).isoformat()
    start = "2020-11-01"

    df = fetch_weather(city, start, end)
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"weather_{city}.csv"
    df.to_csv(path)

    print(f"Saved {len(df)} hourly rows to {path}")
    print(df.head())
    print("Missing values per column:")
    print(df.isna().sum())
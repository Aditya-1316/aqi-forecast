"""Step 2: list PM2.5 stations around a city using the OpenAQ v3 API.

Setup (PowerShell, same terminal you run the script in):
    $env:OPENAQ_API_KEY="paste_your_key_here"
Then:
    venv\\Scripts\\python find_stations.py delhi
"""
import os
import sys

import requests

# bbox = min_lon, min_lat, max_lon, max_lat (a box around each city)
CITY_BBOX = {
    "delhi": (76.84, 28.40, 77.35, 28.88),
}

BASE = "https://api.openaq.org/v3"


def get_headers():
    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        sys.exit("OPENAQ_API_KEY is not set in this terminal. See the top of this file.")
    return {"X-API-Key": key}


def find_pm25_locations(city):
    min_lon, min_lat, max_lon, max_lat = CITY_BBOX[city]
    params = {
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "parameters_id": 2,  # 2 = PM2.5 in OpenAQ; the output below shows the parameter names so you can verify
        "limit": 100,
    }
    resp = requests.get(f"{BASE}/locations", headers=get_headers(), params=params, timeout=60)
    resp.raise_for_status()
    return resp.json().get("results", [])


def show(loc):
    first = (loc.get("datetimeFirst") or {}).get("utc", "?")
    last = (loc.get("datetimeLast") or {}).get("utc", "?")
    print(f"Location {loc['id']}: {loc.get('name')}")
    print(f"   data from {first[:10]} to {last[:10]}")
    for s in loc.get("sensors", []):
        p = s.get("parameter", {})
        print(f"   sensor {s['id']}: {p.get('name')} ({p.get('units')})")


if __name__ == "__main__":
    city = sys.argv[1].lower() if len(sys.argv) > 1 else "delhi"
    if city not in CITY_BBOX:
        sys.exit(f"Unknown city '{city}'. Add its bounding box to CITY_BBOX first.")

    locations = find_pm25_locations(city)
    print(f"Found {len(locations)} locations with PM2.5 near {city}\n")
    for loc in locations:
        show(loc)
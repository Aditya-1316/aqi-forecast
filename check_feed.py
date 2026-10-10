"""Check whether the live PM2.5 feed has really stopped, or whether it is something on our side.

Setup (PowerShell, same terminal):
    $env:OPENAQ_API_KEY="paste_your_key_here"
Run:
    venv\\Scripts\\python check_feed.py
"""
import os
import sys
from datetime import datetime, timezone

import requests

BASE = "https://api.openaq.org/v3"
LOCATION_ID = 11603                       # Chandni Chowk
SENSORS = {12236361: "new feed (the one we use live)", 39410: "old feed"}
DELHI_BBOX = "76.84,28.40,77.35,28.88"    # min_lon, min_lat, max_lon, max_lat


def get(path, params=None):
    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        sys.exit("OPENAQ_API_KEY is not set in this terminal.")
    resp = requests.get(f"{BASE}{path}", headers={"X-API-Key": key},
                        params=params, timeout=60)
    if resp.status_code != 200:
        print(f"   (request to {path} returned HTTP {resp.status_code})")
        return {}
    return resp.json()


if __name__ == "__main__":
    print("Current time (UTC):", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))

    print(f"\n1) Newest values at location {LOCATION_ID} (all sensors):")
    latest = get(f"/locations/{LOCATION_ID}/latest").get("results", [])
    rows = sorted(latest, key=lambda r: (r.get("datetime") or {}).get("utc", ""),
                  reverse=True)
    for r in rows[:6]:
        print(f"   sensor {r.get('sensorsId')}: value {r.get('value')} "
              f"at {(r.get('datetime') or {}).get('utc')}")
    if not rows:
        print("   nothing returned")

    print("\n2) PM2.5 sensors, last data time as the API reports it:")
    for sid, label in SENSORS.items():
        res = get(f"/sensors/{sid}").get("results", [])
        if res:
            s = res[0]
            last = (s.get("datetimeLast") or {}).get("utc")
            print(f"   sensor {sid} ({label}): last data {last}")
        else:
            print(f"   sensor {sid} ({label}): no details returned")

    print("\n3) Every Delhi location with PM2.5 and its last data time:")
    locs = get("/locations", {"bbox": DELHI_BBOX, "parameters_id": 2,
                              "limit": 100}).get("results", [])
    for loc in locs:
        print(f"   location {loc['id']}: {loc.get('name')} -> last data "
              f"{(loc.get('datetimeLast') or {}).get('utc')}")
    if not locs:
        print("   nothing returned")
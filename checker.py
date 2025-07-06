from skyfield.api import EarthSatellite, load
from math import sqrt
from datetime import datetime
import re

ts = load.timescale()

def get_norad_id(line1):
    match = re.match(r"1 (\d+)", line1)
    return int(match.group(1)) if match else None

def get_altitude_km(x, y, z):
    R = sqrt(x**2 + y**2 + z**2)
    return R - 6371  # Earth's average radius in km

def get_leo_sat_ids(tle_file):
    leo_ids = []
    with open(tle_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    for i in range(0, len(lines) - 1, 2):
        line1, line2 = lines[i], lines[i+1]
        norad_id = get_norad_id(line1)
        if norad_id is None:
            continue  # Skip invalid NORAD ID lines
        try:
            sat = EarthSatellite(line1, line2, name=str(norad_id), ts=ts)
            t = sat.epoch
            geocentric = sat.at(t)
            x, y, z = geocentric.position.km
            alt = get_altitude_km(x, y, z)
            if alt < 2000:
                leo_ids.append(norad_id)
        except Exception as e:
            print(f"Error with TLE (ID {norad_id}): {e}")
    return sorted(set(leo_ids))

# 🛰️ Run it
leo_norad_ids = get_leo_sat_ids("leo_tles.txt")

with open("leo_norad_ids.txt", "w") as f:
    for sat_id in leo_norad_ids:
        f.write(f"{sat_id}\n")

print(f"✅ Found {len(leo_norad_ids)} LEO satellites.")

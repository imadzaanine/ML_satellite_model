from skyfield.api import EarthSatellite, load
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Load TLEs from file
def read_tles(filepath):
    with open(filepath) as f:
        lines = [line.strip() for line in f if line.strip()]
    return [(lines[i], lines[i+1]) for i in range(0, len(lines), 2)]

# Convert to (lat, lon)
def tle_to_latlon(line1, line2, time):
    sat = EarthSatellite(line1, line2)
    geocentric = sat.at(time)
    subpoint = geocentric.subpoint()
    return subpoint.latitude.degrees, subpoint.longitude.degrees

# Load TLEs
tles = read_tles("tle_data/tle_5.txt")
now = load.timescale().from_datetime(datetime.now(timezone.utc))

positions = []
for l1, l2 in tles:
    try:
        lat, lon = tle_to_latlon(l1, l2, now)
        positions.append((lat, lon))
    except:
        continue

# ✅ Step 2: Plot on 2D Map
plt.figure(figsize=(12, 6))
m = Basemap(projection='mill', resolution='l')
m.drawcoastlines()
m.drawcountries()
m.drawmapboundary(fill_color='lightblue')
m.fillcontinents(color='lightgray', lake_color='lightblue')

# Convert lat/lon to map x/y
lats, lons = zip(*positions)
x, y = m(lons, lats)

m.scatter(x, y, s=5, color='red', label='Satellites')
plt.title("📡 Satellite Ground Tracks")
plt.legend()
plt.tight_layout()
plt.show()

from skyfield.api import EarthSatellite, load, wgs84
import plotly.graph_objects as go

import numpy as np

# Load time
ts = load.timescale()
t = ts.now()

# Parse TLEs again
def parse_tles(filepath):
    with open(filepath) as f:
        lines = [line.strip() for line in f if line.strip()]
    satellites = []
    for i in range(0, len(lines) - 1, 2):
        if lines[i].startswith('1 ') and lines[i+1].startswith('2 '):
            sat = EarthSatellite(lines[i], lines[i+1], 'SAT', ts)
            satellites.append(sat)
    return satellites

# Get lat/lon/alt
def get_satellite_positions(satellites, time):
    lats, lons = [], []
    for sat in satellites:
        geocentric = sat.at(time)
        subpoint = wgs84.subpoint(geocentric)
        lats.append(subpoint.latitude.degrees)
        lons.append(subpoint.longitude.degrees)
    return lats, lons

# Load and convert
sats = parse_tles('tle_data/tle_5.txt')
lats, lons = get_satellite_positions(sats, t)


fig = go.Figure()

# Add satellite points
fig.add_trace(go.Scattergeo(
    lon = lons,
    lat = lats,
    mode = 'markers',
    marker = dict(size=4, color='red'),
    name = 'Satellites'
))

# Globe style
fig.update_geos(
    projection_type="orthographic",
    showland=True, landcolor="rgb(230, 230, 230)",
    showcountries=True, countrycolor="black",
    showocean=True, oceancolor="lightblue"
)

fig.update_layout(
    title="LEO Satellite Positions (real-time)",
    margin={"r":0,"t":0,"l":0,"b":0}
)

fig.show()
import os
import csv
from skyfield.api import EarthSatellite, load
from datetime import datetime

# Time scale
ts = load.timescale()

# Input/output paths
input_folder = "tle_data"
output_csv = "satellite_dataset.csv"

# Open output CSV
with open(output_csv, "w", newline="") as out_csv:
    writer = csv.writer(out_csv)
    writer.writerow([
        "norad_id", "timestamp", 
        "x_km", "y_km", "z_km", 
        "vx_km_s", "vy_km_s", "vz_km_s",
        "sidereal_time_hours", "earth_rotation_angle_rad"
    ])

    for filename in os.listdir(input_folder):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(input_folder, filename)
        norad_id = int(filename.split("_")[1].split(".")[0])  # from filename

        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        for i in range(0, len(lines) - 1, 2):
            line1, line2 = lines[i], lines[i + 1]

            try:
                sat = EarthSatellite(line1, line2, str(norad_id), ts)
                t = sat.epoch

                geocentric = sat.at(t)
                x, y, z = geocentric.position.km
                vx, vy, vz = geocentric.velocity.km_per_s

                # Correct: get sidereal time from the Time object
                sidereal_time = t.gast  # Greenwich Apparent Sidereal Time (in hours)
                earth_rotation_angle = t.gmst * (3.141592653589793 / 12.0)  # GMST in radians

                writer.writerow([
                    norad_id, t.utc_datetime().isoformat(), 
                    x, y, z,
                    vx, vy, vz,
                    sidereal_time, earth_rotation_angle
                ])

            except Exception as e:
                print(f"❌ Error in {filename} line {i}: {e}")

from spacetrack import SpaceTrackClient
from datetime import datetime, timedelta, UTC
import os
import time

# Authenticate
st = SpaceTrackClient(identity='imad.zaanine@gmail.com', password='imadzaanineomo1234')

# Load NORAD IDs
with open("leo_norad_ids.txt", "r") as f:
    norad_ids = [int(line.strip()) for line in f if line.strip().isdigit()]

# Use longer date range for better coverage (e.g. 90 days)
end = datetime.now(UTC)
start = end - timedelta(days=1826)
date_range = f"{start.date()}--{end.date()}"

# Output folders
os.makedirs("tle_data", exist_ok=True)
missing_ids = []

# Download for each satellite
for idx, norad_id in enumerate(norad_ids):
    try:
        print(f"📡 {idx+1}/{len(norad_ids)} - NORAD ID {norad_id}")
        lines = list(st.gp_history(
            norad_cat_id=norad_id,
            epoch=date_range,
            format='tle',
            iter_lines=True
        ))

        if not lines:
            print(f"⚠️ No TLEs found for NORAD ID {norad_id}")
            missing_ids.append(norad_id)
            continue

        with open(f"tle_data/tle_{norad_id}.txt", "w") as f:
            f.write('\n'.join(lines) + '\n')

        print(f"✅ Saved {len(lines)//2} TLEs")
        time.sleep(1)

    except Exception as e:
        print(f"❌ Failed for {norad_id}: {e}")
        missing_ids.append(norad_id)

# Save missing/empty log
if missing_ids:
    with open("missing_ids.txt", "w") as f:
        for mid in missing_ids:
            f.write(f"{mid}\n")

print(f"\n🎯 Done. Missing/empty files: {len(missing_ids)}")

import requests

# Fill in your Space-Track credentials
USERNAME = 'imad.zaanine@gmail.com'
PASSWORD = 'imadzaanineomo1234'

# Start a session
session = requests.Session()

# Set user-agent headers
session.headers.update({
    'User-Agent': 'Mozilla/5.0',
    'Accept': 'application/json'
})

# Login URL and data
login_url = 'https://www.space-track.org/ajaxauth/login'
login_data = {
    'identity': USERNAME,
    'password': PASSWORD
}

# Log in to Space-Track
resp = session.post(login_url, data=login_data)

if 'Set-Cookie' not in resp.headers and 'application/json' not in resp.headers.get('Content-Type', ''):
    print(" Login failed. Check username and password.")
    exit()

print(" Logged in successfully.")

# Correct query URL — only uses valid filters for tle_latest
tle_url = (
    'https://www.space-track.org/basicspacedata/query/class/tle_latest/'
    'ORDINAL/1/limit/1000/OBJECT_TYPE/PAYLOAD/orderby/NORAD_CAT_ID/format/tle'
)

# Get the TLE data
response = session.get(tle_url)

if response.status_code == 200:
    tle_data = response.text
    with open("leo_tles.txt", "w") as file:
        file.write(tle_data)
    print(" TLEs saved to 'leo_tles.txt'")
elif response.status_code == 204:
    print("No TLEs returned (204 No Content). Try changing filters.")
else:
    print(f" Request failed with status {response.status_code}: {response.text}")

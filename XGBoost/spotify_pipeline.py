import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import os
from dotenv import load_dotenv

# -------------------------
# CONFIG
# -------------------------
load_dotenv()

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError("❌ Spotify credentials not set properly")

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )
)

# -------------------------
# SONG LIST
# -------------------------
songs = [
    "Shape of You Ed Sheeran",
    "Blinding Lights The Weeknd",
    "Tum Hi Ho Arijit Singh",
    "Closer Chainsmokers",
    "Love Yourself Justin Bieber",
    "Believer Imagine Dragons",
    "Perfect Ed Sheeran",
    "Senorita Shawn Mendes",
    "Someone Like You Adele",
    "Let Me Down Slowly Alec Benjamin"
]

rows = []

# -------------------------
# HELPER FUNCTIONS
# -------------------------

def get_era(year):
    year = int(year)
    if year < 2010:
        return "2000_2010"
    elif year < 2020:
        return "2010_2020"
    else:
        return "After_2020"

def popularity_bucket(pop):
    if pop < 40:
        return "low"
    elif pop < 70:
        return "medium"
    else:
        return "high"

# -------------------------
# MAIN LOOP
# -------------------------
for song in songs:
    try:
        result = sp.search(q=f"track:{song}", type="track", limit=1)

        if len(result["tracks"]["items"]) == 0:
            print(f"Not found: {song}")
            continue

        track = result["tracks"]["items"][0]

        title = track["name"]
        artist = track["artists"][0]["name"]
        popularity = track["popularity"]
        release_year = track["album"]["release_date"][:4]

        # -------------------------
        # DERIVED FEATURES
        # -------------------------
        era = get_era(release_year)
        pop_bucket = popularity_bucket(popularity)

        # simple approximations (temporary)
        energy = round(popularity / 100, 3)
        valence = round(popularity / 100, 3)
        tempo = 100 + (popularity % 60)  # fake variation

        rows.append({
            "title": title,
            "artist": artist,

            # 🔥 CORE METADATA
            "popularity": popularity,
            "release_year": release_year,
            "era": era,
            "popularity_bucket": pop_bucket,

            # 🔥 APPROX FEATURES (temporary)
            "energy": energy,
            "valence": valence,
            "tempo": tempo
        })

        print(f"Done: {song}")

        time.sleep(0.2)

    except Exception as e:
        print(f"Error with {song}: {e}")

# -------------------------
# SAVE DATASET
# -------------------------
df = pd.DataFrame(rows)

df.to_csv("spotify_dataset.csv", index=False)

print("\n✅ Dataset created:", df.shape)
print(df.head())
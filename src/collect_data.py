import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
    scope="user-read-recently-played"
))

recent = sp.current_user_recently_played(limit=5)

for item in recent['items']:
    track = item['track']
    preview = track['preview_url']
    if preview:
        print(f"Vibe Found! {track['name']} Preview: {preview}")
    else:
        print(f"No preview for {track['name']} (Typical for some artists)")
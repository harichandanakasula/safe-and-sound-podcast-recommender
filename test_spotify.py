import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_id = "a22758c057ba46c58d92710c1c4aafbc"
client_secret = "0d712dfb4dcc42e995923fdc043f1db3"

try:
    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
    sp = spotipy.Spotify(auth_manager=auth_manager)
    
    # Test with a simple search
    results = sp.search(q='podcast', type='show', limit=1)
    print("SUCCESS! Spotify credentials work!")
    print(f"Found show: {results['shows']['items'][0]['name']}")
    
except Exception as e:
    print(f"ERROR: {e}")
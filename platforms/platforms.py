from abc import ABC, abstractmethod
import os
import json
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm
import math

class Platform(ABC):
    """Abstract class for any music platform."""
    
    @abstractmethod
    def _fetch_library(self):
        """Fetch albums from the platform and saves it to JSON file."""
        pass
    
    @abstractmethod
    def _normalize_album(self, album):
        """Convert raw album data to universal schema."""
        pass

    def save_library(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.abspath(os.path.join(script_dir, ".."))
        save_dir = os.path.abspath(os.path.join(project_dir, "data/libraries/"))
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"{self.platform_name}-recomMLendation.json"
        output_file = os.path.join(save_dir, file_name)

        with open(output_file, "w") as f:
            json.dump(self.library, f, indent=4)

class Spotify(Platform):
    def __init__(self, client_id, client_secret, redirect_uri):
        self.sp             =       spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id       =   client_id,
                client_secret   =   client_secret,
                redirect_uri    =   redirect_uri,
                scope           =   "user-library-read"
        ))

        self.platform_name      =       "Spotify"
        self.library            =       self._fetch_library()

    def _fetch_library(self):
        albums = []
        limit = 50
        reach = limit
        results = self.sp.current_user_saved_albums(limit=limit)
        num_albums = results['total']
        num_pages = math.ceil(num_albums / limit)
        for _ in tqdm(range(num_pages), desc="Going through albums...", leave=False):
            for item in results['items']:
                album = self._normalize_album(item['album'])
                albums.append(album)

            if results['next']:
                results = self.sp.next(results)
            else:
                break

        return albums

    def _normalize_album(self, album):
        album_id = album['id']
        tracks = self.sp.album_tracks(album_id)
        normalized_album = {
            "name": album['name'],
            "artist": ", ".join([artist['name'] for artist in album['artists']]),
            "release_date": album['release_date'],
            "tracks": [track['name'] for track in tracks['items']],
            "genres": []    # fill later from MusicBrainz
        }
        return normalized_album


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=str, required=True, help="Spotify's client ID.")
    parser.add_argument('--client_secret', type=str, required=True, help="Spotify's client's secret.")
    parser.add_argument('--redirect_uri', type=str, required=True, help="Spotify's redirect URI. ")
    args = parser.parse_args()
    client_id, client_secret, redirect_uri = args.client_id, args.client_secret, args.redirect_uri

    spotify = Spotify(client_id, client_secret, redirect_uri)
    spotify.save_library()

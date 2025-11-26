from abc import ABC, abstractmethod
import os, json, spotipy, math, logging
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm
import musicbrainzngs as mb
from musicbrainzngs import WebServiceError
from typing import List, Dict, Any

logging.basicConfig(filename='errors.log',
                    filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.ERROR)

class Library(ABC):
    """Abstract class for any music platform."""

    def __init__(self):
        self.platform   =   None
        self.library    =   []
    
    @abstractmethod
    def _fetch_library(self) -> List[Dict[str, Any]]:
        """Fetch albums from the platform and saves it to JSON file."""
        pass
    
    @abstractmethod
    def _normalize_album(self, album: Dict[str, Any]) -> Dict[str, Any]:
        """Convert raw album data to universal schema."""
        pass

    def _save_library(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.abspath(os.path.join(script_dir, "data/"))
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"{self.platform}-recomMLendation.json"
        output_file = os.path.join(save_dir, file_name)

        with open(output_file, "w") as f:
            json.dump(self.library, f, indent=4)

class Spotify(Library):

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        super().__init__()
        self.sp             =       spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id       =   client_id,
                client_secret   =   client_secret,
                redirect_uri    =   redirect_uri,
                scope           =   "user-library-read"
        ))

        self.platform           =       "Spotify"
        self.limit              =       50
        self.library            =       self._fetch_library()
        self._save_library()

    def _fetch_library(self) -> List[Dict[str, Any]]:
        albums = []
        results = self.sp.current_user_saved_albums(limit=self.limit)
        num_albums = results['total']
        num_pages = math.ceil(num_albums / self.limit)
        for _ in tqdm(range(num_pages), desc="Going through albums...", leave=False):
            for item in results['items']:
                album = self._normalize_album(item['album'])
                albums.append(album)

            if results['next']:
                results = self.sp.next(results)
            else:
                break

        return albums

    def _normalize_album(self, album: Dict[str, Any]) -> Dict[str, Any]:
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


class MusicBrainz(Library):

    def __init__(self, app_name: str, app_version: str, email: str, local_library_path: str):
        super().__init__()
        mb.set_useragent(app_name, app_version, email)

        with open(local_library_path, 'r') as f:
            self.local_library = json.load(f)

        self.platform           =       "MusicBrainz"
        self.library            =       self._fetch_library()
        self._save_library()

    def _fetch_library(self) -> List[Dict[str, Any]]:
        library = []
        for album in tqdm(self.local_library, desc = "Enriching library with MusicBrainz...", leave = False):
            artist, name = album.get("artist"), album.get("name")
            album_data = {}
            try:
                result = mb.search_releases(artist=artist, release=name, limit=1)

                if result["release-list"]:
                    release = result["release-list"][0]
                    album_data = self._normalize_album(release).copy()
                    album_data["source"] = "MusicBrainz"

                else:
                    album_data = album.copy()
                    album_data["source"] = "Spotify"

            except WebServiceError as e:
                album_data = album.copy()
                album_data["source"] = "Spotify"
                logging.error(f"Failed to retrieve MusicBrainz data for {artist}-{name}: {e}", exc_info=True)

            library.append(album_data)

        return library

    def _normalize_album(self, album: Dict[str, Any]) -> Dict[str, Any]:
        return album

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--app_name', type=str, required=True, help="Name of the app.")
    parser.add_argument('--app_version', type=str, required=True, help="Version of the app.")
    parser.add_argument('--email', type=str, required=True, help="Email linked to project.")
    parser.add_argument('--library_path', type=str, required=True, help="Path of the local library.")
    args = parser.parse_args()
    app_name, app_version, email, library_path = args.app_name, args.app_version, args.email, os.path.abspath(args.library_path)

    musicbrainz = MusicBrainz(app_name, app_version, email, library_path)

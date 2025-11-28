from abc import ABC, abstractmethod
import os, json, spotipy, math, logging, pickle
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
    def _fetch_library(self) -> None:
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

        file_name = f"{self.platform}-recomMLendation.pkl"
        output_file = os.path.join(save_dir, file_name)
        with open(output_file, "wb") as f:
            pickle.dump(self.library, f)

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
        self._fetch_library()

    def _fetch_library(self) -> None:
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

        self.library = albums
        self._save_library()

    def _normalize_album(self, album: Dict[str, Any]) -> Dict[str, Any]:
        album_id = album['id']
        tracks = self.sp.album_tracks(album_id)
        normalized_album = {
            "title": album['name'],
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
        self._fetch_library()

    def _fetch_library(self) -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.abspath(os.path.join(script_dir, "data/"))
        pickle_file = os.path.join(save_dir, f"{self.platform}-recomMLendation.pkl")
        if os.path.exists(pickle_file):
            with open(pickle_file, "rb") as f:
                library = pickle.load(f)
        else:
            library = []
        existing_albums = {(album["artist"], album["name"]) for album in library}

        for album in tqdm(self.local_library, desc = "Enriching library with MusicBrainz...", leave = False):
            artist, name = album.get("artist"), album.get("name")
            key = (artist, name)
            if key in existing_albums:
                continue

            album_data = {}
            try:
                result = mb.search_releases(artist=artist, release=name, limit=1)

                if result["release-list"]:
                    release = result["release-list"][0]
                    album_data = self._normalize_album(release).copy()

                else:
                    album_data = album.copy()
                    album_data["source"] = "Spotify"

            except WebServiceError as e:
                album_data = album.copy()
                album_data["source"] = "Spotify"
                logging.error(f"Failed to retrieve MusicBrainz data for {artist}-{name}: {e}", exc_info=True)

            library.append(album_data)
            existing_albums.add(key)

        self.library = library
        self._save_library()

    def _normalize_album(self, album: Dict[str, Any]) -> Dict[str, Any]:
        norm = {}

        norm["id"] = album.get("id")
        norm["title"] = album.get("name") or album.get("title")
        norm["status"] = album.get("status")
        norm["language"] = album.get("text-representation", {}).get("language")
        norm["date"] = album.get("date")
        norm["source"] = "MusicBrainz"

        artist_credit = album.get("artist-credit", [])
        artist_names, artist_ids = [], []

        for ac in artist_credit:
            if isinstance(ac, dict):
                artist = ac.get("artist", {})
                name = artist.get("name") or ac.get("name")
                artist_id = artist.get("id")
            elif isinstance(ac, str):
                name = ac
                artist_id = None
            else:
                name = None
                artist_id = None

            if name:
                artist_names.append(name)
            if artist_id:
                artist_ids.append(artist_id)

        norm["artist"] = ", ".join(artist_names) if artist_names else None
        norm["artist_id"] = ", ".join(artist_ids) if artist_ids else None

        events = album.get("release-event-list", [])
        if events:
            area = events[0].get("area", {})
            norm["area_id"] = area.get("id")
            norm["area_name"] = area.get("name")
        else:
            norm["area_id"] = None
            norm["area_name"] = None

        labels = album.get("label-info-list", [])
        if labels:
            lbl = labels[0].get("label", {})
            norm["label_id"] = lbl.get("id")
            norm["label_name"] = lbl.get("name")
        else:
            norm["label_id"] = None
            norm["label_name"] = None

        medium_list = album.get("medium-list", [])
        norm["medium_formats"] = [m.get("format") for m in medium_list if m.get("format")]

        return norm

    def add_album(self, name: str, artist: str) -> None: 
        '''Search MusicBrainz for an album by artist and name, then add it to the library.'''
        if (artist, name) in {(album.get("artist"), album.get("title")) for album in self.library}:
            print(f"{artist} – {title} is already in the library.")
            return

        album_data = {}
        try:
            result = mb.search_releases(artist=artist, release=name, limit=1)
            
            if result["release-list"]:
                release = result["release-list"][0]
                album_data = self._normalize_album(release).copy()
                album_data["source"] = "MusicBrainz"
                self.library.append(album_data)
                self._save_library()

            else:
                print(f"No release found on MusicBrainz for {artist} – {name}.")

        except WebServiceError as e:
            logging.error(f"Failed to retrieve MusicBrainz data for {artist}-{name}: {e}", exc_info=True)
            print(f"MusicBrainz lookup failed for {artist} – {name}: {e}")

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
    musicbrainz.add_album("The Rainbow Goblins", "Masayoshi Takanaka")

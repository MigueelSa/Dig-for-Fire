from abc import ABC, abstractmethod
import os, json, spotipy, math, logging, pickle, re
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

    class Album(ABC):
        """Abstract base album class."""
        @abstractmethod
        def _normalize_album(self) -> Dict[str, Any]:
            """Convert raw album data to universal schema."""
            pass

        @abstractmethod
        def _clean_tags(self, tags: list[str]) -> list[str]:
            """Convert tags into musical genres"""
            pass

    def __init__(self):
        self.platform: str | None = None
        self.library: List[Dict[str, Any]] = []
    
    @abstractmethod
    def _fetch_library(self) -> None:
        """Fetch albums from the platform and saves it to JSON file."""
        pass

    def _save_library(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.abspath(os.path.join(script_dir, "../data/"))
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

    class Album(Library.Album):
        def __init__(self, data: Dict[str, Any], sp_client: spotipy.Spotify):
            self.data = data
            self.sp = sp_client

        def _normalize_album(self) -> Dict[str, Any]:
            album = self.data
            album_id = album['id']
            tracks = self.sp.album_tracks(album_id)
            normalized_album = {
                "title": album['name'],
                "artist": [artist['name'] for artist in album['artists']],
                "release_date": album['release_date'],
                "tracks": [track['name'] for track in tracks['items']],
                "genres": []    # fill later from MusicBrainz
            }
            return normalized_album

        def _clean_tags(self, tags: list[str]) -> list[str]:
            return tags

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
                album = self.Album(item['album'], self.sp)
                normalized_album = album._normalize_album()
                albums.append(normalized_album)

            if results['next']:
                results = self.sp.next(results)
            else:
                break

        self.library = albums
        self._save_library()


class MusicBrainz(Library):

    class Album(Library.Album):
        def __init__(self, data: Dict[str, Any]):
            self.data           =       data

            # release
            release             =       self.data.get("release", {})
            self.id             =       release.get("id")       
            self.status         =       release.get("status")
            self.language       =       release.get("text-representation", {}).get("language")
            self.barcode        =       release.get("barcode")
            self.artwork        =       release.get("cover-art-archive", {}).get("artwork")
            # artist-credit
            artist_credit       =       release.get("artist-credit") or []
            self.artist         =       [artist.get("artist", {}).get("name") for artist in artist_credit if isinstance(artist, dict)]
            self.countries      =       [artist.get("artist", {}).get("country") for artist in artist_credit if isinstance(artist, dict) and artist.get("artist", {}).get("country")]
            # release-group
            release_group       =       release.get("release-group")
            self.title          =       release_group.get("title")
            self.type           =       release_group.get("type")
            self.date           =       release_group.get("first-release-date")
            tags_list           =       release_group.get("tag-list") or []
            self.tags           =       self._clean_tags([tag.get("name") for tag in tags_list])
            # medium-list
            medium_list         =       release.get("medium-list") or []
            medium              =       medium_list[0] if len(medium_list) > 0 else {}
            self.track_count    =       medium.get("track-count")


        def _normalize_album(self) -> Dict[str, Any]:
            album = {}
    
            album["id"], album["status"], album["language"], album["barcode"], album["artwork"] = self.id, self.status, self.language, self.barcode, self.artwork
            album["artist"], album["countries"] = self.artist, self.countries
            album["title"], album["type"], album["date"], album["tags"] = self.title, self.type, self.date, self.tags
            album["track-count"] = self.track_count

            return album

        def _clean_tags(self, tags: list[str]) -> list[str]:
            cleaned = []
            for t in tags:
                tag = t.lower().strip()

                is_not_domain = "." not in tag
                has_no_colon = ":" not in tag
                has_no_numbers = not re.search(r"\d", tag)

                if is_not_domain and has_no_colon and has_no_numbers:
                    cleaned.append(tag)
                elif re.fullmatch(r"(?:\d{2}|\d{4})s", tag):
                    cleaned.append(tag)

            return cleaned

    def __init__(self, app_name: str, app_version: str, email: str, local_library_path: str):
        super().__init__()
        mb.set_useragent(app_name, app_version, email)

        with open(local_library_path, 'r') as f:
            self.local_library = json.load(f)

        self.platform           =       "MusicBrainz"
        self._fetch_library()

    def _fetch_library(self) -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.abspath(os.path.join(script_dir, "../data/"))
        pickle_file = os.path.join(save_dir, f"{self.platform}-recomMLendation.pkl")
        if os.path.exists(pickle_file):
            with open(pickle_file, "rb") as f:
                library = pickle.load(f)
        else:
            library = []
        existing_albums = {(tuple(album["artist"]), album["title"].lower()) for album in library}

        for album in tqdm(self.local_library, desc = "Enriching library with MusicBrainz...", leave = False):
            artist, name = album.get("artist"), album.get("title")
            key = (tuple(artist), name.lower())
            if key in existing_albums:
                continue

            album_data = {}
            try:
                result = mb.search_releases(artist=artist, release=name, limit=1)

                if result["release-list"]:
                    release = result["release-list"][0]
                    release_id = release["id"]
                    full = mb.get_release_by_id(release_id, includes=["tags", "release-groups", "artist-credits", "media"])
                    new_album = self.Album(full)
                    album_data = new_album._normalize_album().copy()
                    album_data["source"] = "MusicBrainz"

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


    def add_album(self, title: str, artist: str) -> None: 
        '''Search MusicBrainz for an album by artist and name, then add it to the library.'''
        if title in [album.get("title") for album in self.library]:
            print(f"{artist} – {title} is already in the library.")
            return

        album_data = {}
        try:
            result = mb.search_releases(artist=artist, release=title, limit=1)
            
            if result["release-list"]:
                release = result["release-list"][0]
                release_id = release["id"]
                full = mb.get_release_by_id(release_id, includes=["tags", "release-groups", "artist-credits", "media"])
                new_album = self.Album(full)
                album_data = new_album._normalize_album().copy()
                album_data["source"] = "MusicBrainz"
                self.library.append(album_data)
                self._save_library()

            else:
                print(f"No release found on MusicBrainz for {artist} – {title}.")

        except WebServiceError as e:
            logging.error(f"Failed to retrieve MusicBrainz data for {artist}-{title}: {e}", exc_info=True)
            print(f"MusicBrainz lookup failed for {artist} – {title}: {e}")

if __name__ == "__main__":
    import argparse, time
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--app_name', type=str, required=True, help="Name of the app.")
    parser.add_argument('--app_version', type=str, required=True, help="Version of the app.")
    parser.add_argument('--email', type=str, required=True, help="Email linked to project.")
    parser.add_argument('--library_path', type=str, required=True, help="Path of the local library.")
    args = parser.parse_args()

    app_name, app_version, email, library_path = args.app_name, args.app_version, args.email, os.path.abspath(args.library_path)

    musicbrainz = MusicBrainz(app_name, app_version, email, library_path)
    musicbrainz.add_album("The Rainbow Goblins", "Masayoshi Takanaka")

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=str, required=True, help="Spotify's client ID.")
    parser.add_argument('--client_secret', type=str, required=True, help="Spotify's client's secret.")
    parser.add_argument('--redirect_uri', type=str, required=True, help="Spotify's redirect URI. ")
    args = parser.parse_args()
    client_id, client_secret, redirect_uri = args.client_id, args.client_secret, args.redirect_uri

    spotify = Spotify(client_id, client_secret, redirect_uri)
    '''

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Script ran in {elapsed:.2f} seconds")

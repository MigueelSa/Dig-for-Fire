from abc import ABC, abstractmethod
import os, json, spotipy, math, logging, pickle, re
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm
import musicbrainzngs as mb
from musicbrainzngs import WebServiceError

from tags.tags import Tags
from models.models import AlbumData, LibraryData
from utils.paths import output_path


class Library(ABC):
    """Abstract class for any music platform."""

    def __init__(self):
        self.platform: str | None               =   None
        self.library: LibraryData               =   []
        self.local_library: LibraryData | None  =   None

        self.save_dir                           =   output_path("data")

    @abstractmethod
    def _fetch_library(self, **kwargs) -> None:
        """Fetch albums from the platform and saves it to JSON file."""
        pass

    def _save_library(self, write_json=True):
        save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)

        if write_json:
            file_name = f"{self.platform}-Dig-for-Fire.json"
            output_file = os.path.join(save_dir, file_name)
            with open(output_file, "w") as f:
                json.dump(self.library, f, indent=4)

        file_name = f"{self.platform}-Dig-for-Fire.pkl"
        output_file = os.path.join(save_dir, file_name)
        tmp_file = output_file + ".tmp"

        with open(tmp_file, "wb") as f:
            pickle.dump(self.library, f)
        os.replace(tmp_file, output_file)

    def _load_library(self, library_path: str) -> LibraryData:
        with open(library_path) as f:
            library = json.load(f)
        return library

    def _canonical_album(self, album: AlbumData) -> tuple[str, tuple[str], str, str | None]:
            title = re.sub(r"\s+", " ", album["title"].lower().strip())
            title = re.sub(r"\(.*?\)|\[.*?\]|deluxe|remaster(ed)?", "", title)
            artist = tuple(sorted(re.sub(r"\s+", " ", a.lower().strip()) for a in album["artist"] if a))
            date = album.get("date")
            album_id = album.get("id")

            return title, artist, date, album_id

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

    def _feed_release(self, data: AlbumData) -> AlbumData:
        # id
        release_id      =   data.get("id")
        # artist
        artist_credit   =   data.get("artists") or []
        artist          =   [artist.get("name") for artist in artist_credit if isinstance(artist, dict)]
        # album
        title           =   data.get("name", "")
        date            =   data.get("release_date", "")
        genres, tags    =   [], []

        album = {
                "id": release_id,
                "title": title,
                "artist": artist,
                "date": date,
                "genres": genres,
                "tags": tags,
                "source": self.platform
                }

        return album

    def _fetch_library(self, **kwargs) -> None:
        albums = []
        results = self.sp.current_user_saved_albums(limit=self.limit)
        num_albums = results['total']
        num_pages = math.ceil(num_albums / self.limit)
        for _ in tqdm(range(num_pages), desc="Going through albums...", leave=False):
            for item in results['items']:
                album = self._feed_release(item["album"])
                albums.append(album)

            if results['next']:
                results = self.sp.next(results)
            else:
                break

        self.library = albums
        self._save_library()




class MusicBrainz(Library):

    def __init__(self, app_name: str, app_version: str, email: str):
        super().__init__()
        mb.set_useragent(app_name, app_version, email)

        self.platform           =       "MusicBrainz"
        self.skipped_albums     =       set()
        self.tags               =       Tags()

    def _feed_release(self, data: AlbumData) -> AlbumData:
        """
        Feed release data from MusicBrainz into AlbumData structure.
        
        :param self: Instance of the MusicBrainz class.
        :param data: Release data from MusicBrainz.
        :type data: AlbumData
        :return: AlbumData populated with relevant information.
        :rtype: AlbumData
        """
        # release
        release                 =   data.get("release", {})   
        release_id              =   release.get("id")
        # artist
        artist_credit           =   release.get("artist-credit") or []
        artist                  =   [artist.get("artist", {}).get("name") for artist in artist_credit if isinstance(artist, dict)]
        # album
        release_group           =   release.get("release-group") or {}
        title                   =   release_group.get("title", "")
        date                    =   release_group.get("first-release-date", "")
        tags_list               =   release_group.get("tag-list") or []
        tag_names               =   [tag.get("name") for tag in tags_list]
        genres, tags            =   self.tags.genres_tags(tag_names)
        decade                  =   self.tags.get_decade(date)
        if decade is not None:
            tags.append(decade)

        album = {
                "id": release_id,
                "title": title,
                "artist": artist,
                "date": date,
                "genres": genres, # distance_dict
                "tags": tags,
                "source": self.platform
                }

        return album
    

    def fetch_library(self, local_library_path: str, batch_size: int = 50, **kwargs) -> None:
        self.local_library = self._load_library(local_library_path)

        pickle_file = os.path.join(self.save_dir, f"{self.platform}-Dig-for-Fire.pkl")

        if os.path.exists(pickle_file):
            with open(pickle_file, "rb") as f:
                self.library = pickle.load(f)
        else:
            self.library = []

        self._fetch_library(batch_size=batch_size, **kwargs)
        self._save_library()


    def _fetch_library(self, batch_size: int = 50, **kwargs) -> None:
        # this will still be slow on a rerun due to musicbrainz fetching albums and artists with a slight different name from spotify like albums with deluxe and characters like ^...
        assert self.local_library is not None
        library = self.library
        existing_albums = {self._canonical_album(album) for album in library}

        for ialbum, album in enumerate(tqdm(self.local_library, desc = "Enriching library with MusicBrainz...", leave = False, mininterval=5.0)):
            key = self._canonical_album(album)
            if key in existing_albums:
                continue
            artist, name = album.get("artist", []), album.get("title", "")
            artist = " AND ".join(artist) if isinstance(artist, list) else artist

            try:
                result = mb.search_releases(artist=artist, release=name, limit=1)

                if result["release-list"]:
                    release = result["release-list"][0]
                    release_id = release["id"]
                    full = mb.get_release_by_id(release_id, includes=["tags", "release-groups", "artist-credits", "media"])
                    album_data = self._feed_release(full)
                else:
                    album_data = album.copy()
                    album_data["source"] = "Spotify"

            except WebServiceError as e:
                album_data = album.copy()
                album_data["source"] = "Spotify"
                logging.error(f"Failed to retrieve MusicBrainz data for {artist}-{name}: {e}", exc_info=True)

            library.append(album_data)
            existing_albums.add(key)

            if (ialbum+1) % batch_size == 0:
                self.library = library
                self._save_library(write_json=False)

        self.library = library


    def add_album(self, title: str, artist: str) -> None: 
        '''Search MusicBrainz for an album by artist and name, then add it to the library.'''
        if title in [album.get("title") for album in self.library]:
            logging.info(f"{artist} – {title} is already in the library.")
            return

        try:
            result = mb.search_releases(artist=artist, release=title, limit=1)
            
            if result["release-list"]:
                release = result["release-list"][0]
                release_id = release["id"]
                full = mb.get_release_by_id(release_id, includes=["tags", "release-groups", "artist-credits", "media"])
                album_data = self._feed_release(full)
                self.library.append(album_data)
                self._save_library()

            else:
                logging.error(f"No release found on MusicBrainz for {artist} – {title}.")

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

    logging.basicConfig(filename='errors.log',
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.ERROR)

    app_name, app_version, email, library_path = args.app_name, args.app_version, args.email, os.path.abspath(args.library_path)

    musicbrainz = MusicBrainz(app_name, app_version, email)
    musicbrainz.fetch_library(library_path)
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

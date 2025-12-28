import os, json, random
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import musicbrainzngs as mb
from typing import Dict, Any


class Recommender:

    class Album():
        def __init__(self, data: Dict[str, Any]):
            self.data           =       data

            # release
            self.release_date   =       self.data.get("first-release-date")       
            self.title          =       self.data.get("title")
            # artist-credit
            artist_credit       =       self.data.get("artist-credit") or []
            self.artist         =       [artist.get("artist", {}).get("name") for artist in artist_credit if isinstance(artist, dict)]
            # tag-list
            tag_list            =       self.data.get("tag-list") or []
            self.tags           =       [tag.get("name") for tag in tag_list if isinstance(tag, dict)]

        def _normalize_album(self) -> Dict[str, any]:
            album = {}
            album["release_date"], album["title"], album["artist"], album["tags"] = self.release_date, self.title, self.artist, self.tags
            return album

    def __init__(self, library_path: str, email: str, app_name: str = "recomMLendation", version: str = "0.1", k: int = 10, limit: int = 50):
        self.library                =   self._load_library(library_path)
        self.library_space_matrix   =   self._library_space_matrix()
        self.taste_vector           =   self._taste_vector()
        self.similarity_matrix      =   cosine_similarity(self.library_space_matrix, self.taste_vector)
        self.fetched_albums         =   self._fetch_albums(app_name, version, email, k, limit)



    def _load_library(self, library_path: str) -> dict[str, any]:
        with open(library_path, "r") as file:
            library = json.load(file)
        return library

    def _album_to_tokens(self, album: dict[str, any]) -> list[str]:
        tokens = []
        tokens.extend(album.get("tags", []))
        if "decade" in album:
            tokens.append(album["decade"])
        return tokens

    def _album_tags(self) -> list[str]:
        docs = [" ".join(self._album_to_tokens(a)) for a in self.library]
        return docs

    def _library_space_matrix(self) -> csr_matrix:
        docs = self._album_tags()
        vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"[^ ]+")
        X = vectorizer.fit_transform(docs)
        return X

    def _taste_vector(self) -> np.ndarray:
        taste = self.library_space_matrix.sum(axis=0)
        taste = np.asarray(taste)
        taste = normalize(taste)
        # now this should be true taste.shape == (1, V)
        return taste

    def _genre_randomizer(self, k: int) -> list:
        docs = self._album_tags()
        tokens = set()
        for album in docs:
            tokens.update(album.split())
        tokens = list(tokens)
        picked_genres = random.sample(tokens, k=k)
        return picked_genres

    def _fetch_albums(self, app_name: str, version: str, email: str, k: int, limit: int) -> list[dict[str, any]]:
        random_genres = self._genre_randomizer(k)
        mb.set_useragent(app_name, version, email)
        fetched_albums = []
        for ig, genre in enumerate(random_genres):
            result = mb.search_release_groups(query=f'tag:{genre} AND primarytype:album', limit=limit)
            releases = result["release-group-list"]
            for release in releases:
                fetched_album = self.Album(release)
                normalized_album = fetched_album._normalize_album()
                fetched_albums.append(normalized_album)
        return fetched_albums
        


if __name__ == "__main__":
    import argparse, time
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--library_path', type=str, required=True, help="Path of the local library.")
    parser.add_argument('--email', type=str, required=True, help="User email address.")
    args = parser.parse_args()

    library, email = os.path.abspath(args.library_path), args.email

    rec = Recommender(library, email)
    fetched_albums = rec.fetched_albums
    print(fetched_albums)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Script ran in {elapsed:.2f} seconds")

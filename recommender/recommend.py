import os, json, random
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import musicbrainzngs as mb
from typing import Dict, Any
import pandas as pd
from tqdm import tqdm

from types.types import AlbumData, LibraryData
from albums.albums import Album


class Recommender:

    def __init__(self, library_path: str, email: str, app_name: str = "recomMLendation", version: str = "0.1", k: int = 5, limit: int = 100, threshold: float = 0.4):
        self.library                                                                =   self._load_library(library_path)
        self.library_space_matrix                                                   =   self._space_matrix(self.library)
        self.taste_vector                                                           =   self._taste_vector()
        self.app_name, self.version, self.email, self.k, self.limit, self.threshold =   app_name, version, email, k, limit, threshold   

    def _feed_release(self, data: AlbumData) -> AlbumData:
        # release
        release         =   data.get("release", {})   
        release_id      =   release.get("id")
        # artist
        artist_credit   =   release.get("artist-credit") or []
        artist          =   [artist.get("artist", {}).get("name") for artist in artist_credit if isinstance(artist, dict)]
        # album
        release_group   =   release.get("release-group") or {}
        title           =   release_group.get("title", "")
        date            =   release_group.get("first-release-date", "")
        tags_list       =   release_group.get("tag-list") or []
        tag_names       =   [tag.get("name") for tag in tags_list]
        genres, tags    =   self.tags.genre_tags(tag_names)
        tags.append(self.tags.get_decade(date))

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

    def _fetch_recommendations(self) -> LibraryData:
        app_name, version, email, k, limit, threshold = self.app_name, self.version, self.email, self.k, self.limit, self.threshold
        kept_albums = []
        with tqdm(total=k, desc="Albums found.", leave = False) as pbar:
            while len(kept_albums) < k:
                fetched_albums = self._fetch_albums(app_name, version, email, k, limit)
                fetched_albums_space_matrix = self._space_matrix(fetched_albums, create_space=False)
                similarity_matrix = cosine_similarity(fetched_albums_space_matrix, self.taste_vector)
                for album, similarity_score in zip(fetched_albums, similarity_matrix.flatten()):
                    if similarity_score >= threshold and album not in [a for a, _ in kept_albums]:
                        kept_albums.append((album, similarity_score))
                        pbar.update(1)
            df = pd.DataFrame(kept_albums, columns=["album", "score"])
            df = df.sort_values("score", ascending=False)
            top_albums = []
            for _, row in df.iterrows():
                album_info = row["album"]
                top_albums.append({
                    "artist": album_info.get("artist"),
                    "title": album_info.get("title"),
                    "release_date": album_info.get("release_date"),
                    "tags": album_info.get("tags"),
                    "score": row["score"]
                })
        return top_albums[:k]

    def recommend(self) -> str:
        top_albums = self._fetch_recommendations()
        output = ""
        for ialbum,  album in enumerate(top_albums):
            output += f"{ialbum+1}. {album['title']} by {', '.join(album['artist'])}. Tags: {', '.join(album['tags'])}. Similarity score: {album['score']}.\n"
        return output

    def _load_library(self, library_path: str) -> LibraryData:
        with open(library_path, "r") as file:
            library = json.load(file)
        return library

    def _album_to_tokens(self, album: AlbumData) -> list[str]:
        tokens = []
        tokens.extend(album.get("tags", []))
        if "decade" in album:
            tokens.append(album["decade"])
        return tokens

    def _album_tags(self, library: LibraryData) -> list[str]:
        docs = [" ".join(self._album_to_tokens(a)) for a in library]
        return docs

    def _space_matrix(self, library: LibraryData, create_space=True) -> csr_matrix:
        docs = self._album_tags(library)
        if create_space:
            self.vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"[^ ]+")
            X = self.vectorizer.fit_transform(docs)
        else:
            X = self.vectorizer.transform(docs)
        return X

    def _taste_vector(self) -> np.ndarray:
        taste = self.library_space_matrix.sum(axis=0)
        taste = np.asarray(taste)
        taste = normalize(taste)
        # now this should be true taste.shape == (1, V)
        return taste

    def _genre_randomizer(self, k: int) -> list:
        # do something that takes the frequency of the genre in the library into account
        docs = self._album_tags(self.library)
        tokens = set()
        for album in docs:
            tokens.update(album.split())
        tokens = list(tokens)
        picked_genres = random.sample(tokens, k=k)
        return picked_genres

    def _fetch_albums(self, app_name: str, version: str, email: str, k: int, limit: int) -> LibraryData:
        random_genres = self._genre_randomizer(k)
        mb.set_useragent(app_name, version, email)
        fetched_albums = []
        for ig, genre in enumerate(random_genres):
            result = mb.search_release_groups(query=f'tag:{genre} AND primarytype:album', limit=limit)
            releases = result["release-group-list"]
            for release in releases:
                fetched_album_dict = self._feed_release(release)
                fetched_albums.append(fetched_album_dict)
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
    recommendations = rec.recommend()
    print(recommendations)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Script ran in {elapsed:.2f} seconds")

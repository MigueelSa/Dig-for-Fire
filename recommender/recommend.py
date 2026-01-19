import logging
import os, json, random
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import musicbrainzngs as mb
import pandas as pd
from tqdm import tqdm
from collections import Counter

from models.models import AlbumData, LibraryData
from libraries.libraries import MusicBrainz
from utils.paths import output_path


class Recommender:

    def __init__(self, library_path: str, email: str, app_name: str = "Dig-for-Fire", app_version: str = "0.1", k: int = 2, limit: int = 1, threshold: float = 0.3):
        self.mb_library                                                                 =   MusicBrainz(app_name, app_version, email)
        self.library                                                                    =   self._load_library(library_path)
        self.genre_space_matrix, self.tag_space_matrix                                  =   self._space_matrix(self.library)
        self.taste_gv, self.taste_tv                                                    =   self._taste_vectors()
        self.app_name, self.app_version, self.email, self.k, self.limit, self.threshold =   app_name, app_version, email, k, limit, threshold   
        self.used_tokens                                                                =   set()
        self.excluded_albums                                                            =   self._excluded_albums()

    def _fetch_recommendations(self) -> LibraryData:
        k, limit, threshold = self.k, self.limit, self.threshold
        kept_albums, kept_ids = [], set()
        with tqdm(total=k, desc="Albums found.", leave = False) as pbar:
            while len(kept_albums) < k:
                random_tokens = self._genre_tag_randomizer(1)
                fetched_albums = self._fetch_albums(random_tokens, limit)
                if not fetched_albums:
                    continue
                fetched_albums_gspace_matrix, fetched_albums_tspace_matrix = self._space_matrix(fetched_albums, create_space=False)
                genre_similarity_projections, tag_similarity_projections = cosine_similarity(fetched_albums_gspace_matrix, self.taste_gv.reshape(1, -1)), cosine_similarity(fetched_albums_tspace_matrix, self.taste_tv.reshape(1, -1))
                similarity_scores = self._similarity_scores(genre_similarity_projections, tag_similarity_projections)
                for album, similarity_score in zip(fetched_albums, similarity_scores.flatten()):
                    album_id = self.mb_library._canonical_album(album)
                    if similarity_score >= threshold and album_id not in kept_ids and album_id not in self.excluded_albums:
                        kept_albums.append((album, similarity_score))
                        kept_ids.add(album_id)
                        self.excluded_albums.add(album_id)
                        pbar.update(1)
            df = pd.DataFrame(kept_albums, columns=["album", "score"])
            df = df.sort_values("score", ascending=False)
            top_albums = []
            for _, row in df.iterrows():
                album_info = row["album"]
                album_dict = {
                    "id": album_info.get("id"),
                    "title": album_info.get("title"),
                    "artist": album_info.get("artist"),
                    "date": album_info.get("date"),
                    "genres": album_info.get("genres", []),
                    "tags": album_info.get("tags", []),
                    "source": album_info.get("source"),
                    "score": row["score"]
                }
                top_albums.append(album_dict)
        return top_albums[:k]

    def _similarity_scores(self, genre_similarity_projections: np.ndarray, tag_similarity_projections: np.ndarray, genre_weight: float = 1.0, tag_weight: float = 0.1) -> np.ndarray:
        genre_scores = genre_similarity_projections.flatten()
        tag_scores = tag_similarity_projections.flatten()
        combined_scores = (genre_weight * genre_scores + tag_weight * tag_scores) / (genre_weight + tag_weight)
        return combined_scores
    
    def recommend(self) -> str:
        top_albums = self._fetch_recommendations()
        self._save_recommendations(top_albums)
        output = ""
        for ialbum,  album in enumerate(top_albums):
            output += f"{ialbum+1}. {album['title']} by {', '.join(album['artist'])}. Genres: {', '.join(album['genres'])}. Tags: {', '.join(album['tags'])}. Similarity score: {album['score']}.\n"
        return output

    def _load_library(self, library_path: str) -> LibraryData:
        with open(library_path, "r") as file:
            library = json.load(file)
        return library

    def _album_genres_tags(self, album: AlbumData) -> tuple[list[str], list[str]]:
        genres = list(album.get("genres", []))
        tags = list(album.get("tags", []))
        genre_set = set(genres)
        for genre in genres:
            parents = self.mb_library.tags.parents.get(genre) or []
            for parent in parents:
                if parent in genre_set:
                    genre_set.remove(parent)
        filtered_genres = list(genre_set)
        return filtered_genres, tags

    def _library_genres_tags(self, library: LibraryData) -> tuple[list[str], list[str]]:
        library_genres, library_tags = [], []
        for album in library:
            genres, tags = self._album_genres_tags(album)
            library_genres.append(" ".join(genres))
            library_tags.append(" ".join(tags))
        return library_genres, library_tags

    def _space_matrix(self, library: LibraryData, create_space=True) -> tuple[csr_matrix, csr_matrix]:
        library_genres, library_tags = self._library_genres_tags(library)
        if create_space:
            self.gvectorizer, self.tvectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"[^ ]+"), TfidfVectorizer(lowercase=False, token_pattern=r"[^ ]+")
            X, Y = self.gvectorizer.fit_transform(library_genres), self.tvectorizer.fit_transform(library_tags)
        else:
            X, Y = self.gvectorizer.transform(library_genres), self.tvectorizer.transform(library_tags)
        return X, Y

    def _taste_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        gv, tv = self.genre_space_matrix.sum(axis=0), self.tag_space_matrix.sum(axis=0)
        gv, tv = np.asarray(gv), np.asarray(tv)
        gv, tv = normalize(gv), normalize(tv)
        return gv, tv

    def _genre_tag_randomizer(self, k: int, genre_weight: float = 1.0, tag_weight: float = 0.1) -> list[tuple[str, str]]:
        # Collect all genres and tags from the library
        all_genres, all_tags = [], []
        for album in self.library:
            all_genres.extend(album.get("genres", []))
            all_tags.extend(album.get("tags", []))
        
        # Count frequencies
        genre_counts, tag_counts = Counter(all_genres), Counter(all_tags)
        
        # Create a weighted pool
        pool = (
        [(g, "genre", c) for g, c in genre_counts.items()] +
        [(t, "tag", c) for t, c in tag_counts.items()]
        )

        if not pool:
            return []

        # Separate items and weights
        items, weights = zip(*[(x[:2], x[2] * (genre_weight if x[1] == "genre" else tag_weight)) for x in pool])
        items, weights = list(items), np.array(weights, dtype=float)

        random_tokens, tokens_set = [], set()
        while len(random_tokens) < k:
            selected = random.choices(items, weights=weights, k=1)[0]
            token, _ = selected
            if token not in self.used_tokens and token not in tokens_set:
                random_tokens.append(selected)
                tokens_set.add(token)
        self.used_tokens.update(token for token, _ in random_tokens)
        return random_tokens
        

    def _fetch_albums(self, tokens: list[tuple[str, str]], limit: int) -> LibraryData:
        fetched_albums = []
        for token, _ in tokens:
            result = mb.search_releases(query=f'tag:{token} AND primarytype:album', limit=limit)
            releases = result.get("release-list", [])
            for release in releases:
                release_id = release.get("id")
                if not release_id:
                    continue
                try:
                    full = mb.get_release_by_id(release_id, includes=["tags", "artist-credits", "release-groups", "media"])
                except mb.ResponseError as e:
                    logging.warning(f"Skipping release {release_id}: {e}")
                    continue
                fetched_album_dict = self.mb_library._feed_release(full)
                fetched_album_id = self.mb_library._canonical_album(fetched_album_dict)
                if fetched_album_id not in self.excluded_albums:
                    fetched_albums.append(fetched_album_dict)
        return fetched_albums
    
    def _save_recommendations(self, recommendations: LibraryData, output_path = output_path("data")) -> None:
        recommendation_history_path = os.path.abspath(os.path.join(output_path, "recommendation-history-Dig-for-Fire.json"))
        history = self._load_recommendations()
        history.extend(recommendations)
        with open(recommendation_history_path, "w") as file:
            json.dump(history, file, indent=4)

    def _load_recommendations(self, input_path = output_path("data", "recommendation-history-Dig-for-Fire.json")) -> LibraryData:
        if not os.path.exists(input_path):
            with open(input_path, "w") as file:
                json.dump([], file)
        with open(input_path, "r") as file:
            recommendations = json.load(file)
        return recommendations
    
    def _excluded_albums(self) -> set[tuple[str, str, str, str | None]]:
        excluded = set()
        recommendation_history = self._load_recommendations()
        for album in recommendation_history:
            album_id = self.mb_library._canonical_album(album)
            excluded.add(album_id)
        for album in self.library:
            album_id = self.mb_library._canonical_album(album)
            excluded.add(album_id)
        return excluded
        


if __name__ == "__main__":
    import argparse, time

    logging.basicConfig(filename='errors.log',
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.ERROR)
    
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
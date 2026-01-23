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
        self.mb_library                                                                     =   MusicBrainz(app_name, app_version, email)
        self.library                                                                        =   self._load_library(library_path)
        self.genre_space_matrix, self.parent_space_matrix, self.tag_space_matrix            =   self._space_matrix(self.library)
        self.taste_gv, self.taste_pv, self.taste_tv                                         =   self._taste_vectors()
        self.app_name, self.app_version, self.email, self.k, self.limit, self.threshold     =   app_name, app_version, email, k, limit, threshold   
        self.used_tokens                                                                    =   set()
        self.excluded_albums                                                                =   self._excluded_albums()
        self.children_genres, self.parent_genres, self.parent_unseen_children, self.tags    =   self._pools()

    def _fetch_recommendations(self) -> LibraryData:
        k, limit, threshold = self.k, self.limit, self.threshold
        kept_albums, kept_ids = [], set()
        with tqdm(total=k, desc="Albums found.", leave = False) as pbar:
            while len(kept_albums) < k:
                random_tokens = self._genre_tag_randomizer(1)
                fetched_albums = self._fetch_albums(random_tokens, limit)
                if not fetched_albums:
                    continue
                fetched_albums_gspace_matrix, fetched_albums_pspace_matrix, fetched_albums_tspace_matrix = self._space_matrix(fetched_albums, create_space=False)
                genre_similarity_projections = cosine_similarity(fetched_albums_gspace_matrix, self.taste_gv)
                parent_similarity_projections = cosine_similarity(fetched_albums_pspace_matrix, self.taste_pv)
                tag_similarity_projections = cosine_similarity(fetched_albums_tspace_matrix, self.taste_tv)
                similarity_scores = self._similarity_scores(genre_similarity_projections, parent_similarity_projections, tag_similarity_projections)
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

    def _similarity_scores(self, genre_similarity_projections: np.ndarray, parent_similarity_projections: np.ndarray, tag_similarity_projections: np.ndarray, genre_weight: float = 0.65, parent_weight: float = 0.25) -> np.ndarray:
        tag_weight = 1.0 - genre_weight - parent_weight
        genre_scores = genre_similarity_projections.flatten()
        parent_scores = parent_similarity_projections.flatten()
        tag_scores = tag_similarity_projections.flatten()
        combined_scores = (genre_weight * genre_scores + parent_weight * parent_scores + tag_weight * tag_scores) / (genre_weight + parent_weight + tag_weight)
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

    def _library_genres_tags(self, library: LibraryData) -> tuple[list[str], list[str], list[str]]:
        library_genres, parent_genres, library_tags = [], [], []
        for album in library:
            genres, tags = self._album_genres_tags(album)
            library_genres.append(" ".join(genres))
            library_tags.append(" ".join(tags))

            parents = self._albums_parents(album)
            parent_genres.append(" ".join(parents))

        return library_genres, parent_genres, library_tags

    def _space_matrix(self, library: LibraryData, create_space=True) -> tuple[csr_matrix, csr_matrix]:
        library_genres, parent_genres, library_tags = self._library_genres_tags(library)
        if create_space:
            self.gvectorizer, self.pvectorizer, self.tvectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"[^ ]+"), TfidfVectorizer(lowercase=False, token_pattern=r"[^ ]+"), TfidfVectorizer(lowercase=False, token_pattern=r"[^ ]+")
            X, Y, Z = self.gvectorizer.fit_transform(library_genres), self.pvectorizer.fit_transform(parent_genres), self.tvectorizer.fit_transform(library_tags)
        else:
            X, Y, Z = self.gvectorizer.transform(library_genres), self.pvectorizer.transform(parent_genres), self.tvectorizer.transform(library_tags)
        return X, Y, Z

    def _taste_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gv, pv, tv = self.genre_space_matrix.sum(axis=0), self.parent_space_matrix.sum(axis=0), self.tag_space_matrix.sum(axis=0)
        gv, pv, tv = np.asarray(gv, dtype=float).reshape(1, -1), np.asarray(pv, dtype=float).reshape(1, -1), np.asarray(tv, dtype=float).reshape(1, -1)
        gv, pv, tv = normalize(gv), normalize(pv), normalize(tv)
        return gv, pv, tv

    def _pools(self) -> tuple[dict[str, int], dict[str, int], dict[str, list[str]], dict[str, int]]:
        all_genres, all_tags = [], []
        for album in self.library:
            all_genres.extend(album.get("genres", []))
            all_tags.extend(album.get("tags", []))
        genre_counts, tag_counts = Counter(all_genres), Counter(all_tags)
        
        children_genres, parent_genres = {}, {}
        parent_children = self.mb_library.tags._get_parents_children()
        for parent, children in parent_children.items():
            for child in children:
                children_genres[child] = genre_counts.get(child, 0)
                parent_genres[parent] = parent_genres.get(parent, 0) + genre_counts.get(child, 0)

        parent_unseen_children = {
            parent: unseen
            for parent, children in parent_children.items()
            if parent_genres.get(parent, 0) > 0
            and (unseen := [child for child in children if children_genres.get(child, 0) == 0])
        }

        tags = dict(tag_counts)

        return children_genres, parent_genres, parent_unseen_children, tags

    def _genre_tag_randomizer(self, k: int, cg_probability: float = 0.6, pg_probability: float = 0.25) -> list[tuple[str, str]]:
        children_genres, parent_genres, parent_unseen_children, tags = self.children_genres, self.parent_genres, self.parent_unseen_children, self.tags

        cg_items, cg_weights = [], []
        for child, weight in children_genres.items():
            if child not in self.used_tokens:
                cg_items.append(child)
                cg_weights.append(weight)

        pg_items, pg_weights = [], []
        for parent, weight in parent_genres.items():
            if parent_unseen_children.get(parent) and parent not in self.used_tokens:
                pg_items.append(parent)
                pg_weights.append(weight)

        t_items, t_weights = [], []
        for tag, weight in tags.items():
            if tag not in self.used_tokens:
                t_items.append(tag)
                t_weights.append(weight)

        random_tokens, tokens_set = [], set()
        while len(random_tokens) < k:
            selected, r = None, random.random()
            if r < cg_probability and cg_items:
                selected = random.choices(cg_items, weights=cg_weights, k=1)[0], "genre"
            elif r < cg_probability + pg_probability and pg_items:
                parent = random.choices(pg_items, weights=pg_weights, k=1)[0]
                selected = random.choices(parent_unseen_children.get(parent, []), k=1)[0], "genre"
            elif t_items:
                selected = random.choices(t_items, weights=t_weights, k=1)[0], "tag"

            if selected is None:
                available_items = [(x, "genre") for x in cg_items + pg_items] + [(x, "tag") for x in t_items]
                if not available_items:
                    break
                selected = random.choice(available_items)

            token, _ = selected
            if token not in self.used_tokens and token not in tokens_set:
                random_tokens.append(selected)
                tokens_set.add(token)
        self.used_tokens.update(token for token, _ in random_tokens)
        return random_tokens
        

    def _fetch_albums(self, tokens: list[tuple[str, str]], limit: int) -> LibraryData:
        fetched_albums = {}
        for token, _ in tokens:
            try:
                result = mb.search_releases(query=f'tag:{token} AND primarytype:album', limit=limit)
                releases = result.get("release-list", [])
            except (mb.ResponseError, mb.NetworkError) as e:
                logging.warning(f"Skipping token {token}: {e}")
                continue

            for release in releases:
                release_id = release.get("id")
                if not release_id:
                    continue
                try:
                    full = mb.get_release_by_id(release_id, includes=["tags", "artist-credits", "release-groups", "media"])
                except (mb.ResponseError, mb.NetworkError) as e:
                    logging.warning(f"Skipping release {release_id}: {e}")
                    continue

                fetched_album_dict = self.mb_library._feed_release(full)
                fetched_album_id = self.mb_library._canonical_album(fetched_album_dict)
                if fetched_album_id not in self.excluded_albums:
                    fetched_albums[fetched_album_id] = fetched_album_dict

        return list(fetched_albums.values())
    
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

    def _albums_parents(self, album: AlbumData) -> list[str]:
        genres = album.get("genres", [])
        parents = set()
        for genre in genres:
            genre_parents = self.mb_library.tags.parents.get(genre) or []
            parents.update(genre_parents)
        return list(parents)
        


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
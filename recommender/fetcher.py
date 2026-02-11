import logging, time, re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import musicbrainzngs as mb
from sklearn.preprocessing import normalize

from embeddings.genre_space import GenreSpace
from embeddings.tag_space import TagSpace
from ml.predictor import Predictor
from models.models import AlbumData, LibraryData
from utils.loading import loading_animation
from recommender.explorer import Explorer
from libraries.libraries import MusicBrainz

class Fetcher:
    def __init__(self, library: dict[str, AlbumData], mb_library: MusicBrainz, genre_embeddings: GenreSpace, tag_embeddings: TagSpace, 
                 recommendation_history: LibraryData, k: int = 2, limit: int = 10, threshold: float = 0.6):
        self.library: LibraryData                   =   library
        self.mb_library: MusicBrainz                =   mb_library
        self.genre_embeddings: GenreSpace           =   genre_embeddings
        self.tag_embeddings: TagSpace               =   tag_embeddings
        self.recommendation_history: LibraryData    =   recommendation_history
        self.k: int                                 =   k
        self.limit: int                             =   limit
        self.threshold: float                       =   threshold
        self.explorer: Explorer                     =   Explorer(library, self.mb_library.tags.roots, self.mb_library.tags.ancestors_children)
        self.taste_gv, self.taste_tv                =   self._taste_vectors()
        self.excluded_albums                        =   self._excluded_albums()
        self.predictor                              =   Predictor(self.recommendation_history, self.library, self.genre_embeddings, self.tag_embeddings, balanced=False)

    def _canonical_album(self, album: AlbumData) -> tuple[str, tuple[str], str, str | None]:
            title = re.sub(r"\s+", " ", album["title"].lower().strip())
            title = re.sub(r"\(.*?\)|\[.*?\]|deluxe|remaster(ed)?", "", title)
            artist = tuple(sorted(re.sub(r"\s+", " ", a.lower().strip()) for a in album["artist"] if a))
            date = album.get("date")
            album_id = album.get("id")

            return title, artist, date, album_id
    
    def _fetch_albums(self, tokens: list[tuple[str, str]], limit: int) -> LibraryData:
        fetched_albums = {}
        for token, _ in tokens:
            try:
                result = mb.search_releases(query=f'tag:{token} AND primarytype:album', limit=limit)
                releases = result.get("release-list", [])
                time.sleep(1)  # throttle: wait 1 second between requests
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
                fetched_album_id = self._canonical_album(fetched_album_dict)
                if fetched_album_id not in self.excluded_albums:
                    fetched_albums[fetched_album_id] = fetched_album_dict

        return list(fetched_albums.values())
    
    def _fetch_recommendations(self) -> LibraryData:
        k, limit, threshold = self.k, self.limit, self.threshold
        kept_albums, kept_ids = [], set()
        albums_looked = 0
        albums_found_message = f"Fetching recommendations. 0/{k} albums found."
        normal_message = albums_found_message + f" {albums_looked} albums looked up."
        message = [normal_message]
        stop = loading_animation(message)
        while len(kept_albums) < k:
            albums_found_message = f"Fetching recommendations. {len(kept_albums)}/{k} albums found."
            normal_message = albums_found_message + f" {albums_looked} albums looked up."
            picky_message = normal_message + " You are really picky!"
            message[0] = normal_message if albums_looked < 25 * self.k else picky_message

            #print("starts here")
            random_tokens = self.explorer._genre_tag_randomizer(1)
            #print("randomizer done")
            fetched_albums = self._fetch_albums(random_tokens, limit)
            #print("fetch done")
            if not fetched_albums:
                continue
            fetched_albums_gspace_matrix, fetched_albums_tspace_matrix = self._space_matrix(fetched_albums)
            #print("space matrix done")

            genre_similarity_projections = cosine_similarity(fetched_albums_gspace_matrix, self.taste_gv)
            tag_similarity_projections = cosine_similarity(fetched_albums_tspace_matrix, self.taste_tv)
            #print("similarity done")
            similarity_scores = self._similarity_scores(genre_similarity_projections, tag_similarity_projections)

            if self.predictor.is_safe:
                album_features = np.hstack([fetched_albums_gspace_matrix, fetched_albums_tspace_matrix])
                album_scores = self.predictor.model.predict_proba(album_features)[:, 1]
                similarity_scores = (1-self.predictor.alpha)*similarity_scores + self.predictor.alpha*album_scores

            for album, similarity_score in zip(fetched_albums, similarity_scores.flatten()):
                album_id = self._canonical_album(album)
                if similarity_score >= threshold and album_id not in kept_ids and album_id not in self.excluded_albums:
                    kept_albums.append((album, similarity_score))
                    kept_ids.add(album_id)
                    self.excluded_albums.add(album_id)
            #print("albums processed")
            albums_looked += len(fetched_albums)
        stop()
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
    
    def _excluded_albums(self) -> set[tuple[str, str, str, str | None]]:
        excluded = set()
        for album in self.recommendation_history:
            album_id = self._canonical_album(album)
            excluded.add(album_id)
        for album in self.library:
            album_id = self._canonical_album(album)
            excluded.add(album_id)
        return excluded
    
    def _space_matrix(self, library: LibraryData) -> tuple[np.ndarray, np.ndarray]:
        g_list, t_list = [], []
        for album in library:
            genre, tag = self.genre_embeddings.get_album_embeddings(album), self.tag_embeddings.get_album_embeddings(album)
            g_list.append(genre)
            t_list.append(tag)
        g_array, t_array = np.vstack(g_list), np.vstack(t_list)
        return g_array, t_array
    
    def _similarity_scores(self, genre_similarity_projections: np.ndarray, tag_similarity_projections: np.ndarray, genre_weight: float = 0.6) -> np.ndarray:
        tag_weight = 1.0 - genre_weight
        genre_scores = genre_similarity_projections.flatten()
        tag_scores = tag_similarity_projections.flatten()
        combined_scores = (genre_weight * genre_scores + tag_weight * tag_scores) / (genre_weight + tag_weight)
        return combined_scores
    
    def _taste_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        gv, tv = None, None

        for album in self.library:
            album_genres, album_tags = self.genre_embeddings.get_album_embeddings(album), self.tag_embeddings.get_album_embeddings(album)
            
            gv_vec = album_genres
            gv = gv_vec if gv is None else gv + gv_vec
        
            tv_vec = album_tags
            tv = tv_vec if tv is None else tv + tv_vec

        gv = normalize(gv.reshape(1, -1)) if gv is not None else np.zeros((1, len(next(iter(self.genre_embeddings.dimension)))))
        tv = normalize(tv.reshape(1, -1)) if tv is not None else np.zeros((1, len(next(iter(self.tag_embeddings.dimension)))))

        return gv, tv
    
    
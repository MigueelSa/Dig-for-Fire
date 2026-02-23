import logging, time, re, requests, logging, json, os, pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import musicbrainzngs as mb
from sklearn.preprocessing import normalize

from digforfire.embeddings.genre_space import GenreSpace
from digforfire.embeddings.tag_space import TagSpace
from digforfire.ml.predictor import Predictor
from digforfire.models.models import AlbumData, LibraryData
from digforfire.utils.loading import loading_animation
from digforfire.recommender.explorer import Explorer
from digforfire.libraries.libraries import MusicBrainz
from digforfire.utils.paths import output_path

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
    
    def _fetch_albums(self, tokens: list[tuple[str, str]], api_key: str, limit: int) -> LibraryData:
        fetched_albums = {}
        for token, token_type in tokens:
            try:
                if token_type == "artist":
                    artist = self._fetch_similar([token], api_key, limit=limit)
                    if not artist:
                        continue
                    result = mb.search_releases(query=f'artist:{artist} AND primarytype:album', limit=limit)
                else:
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
    
    def _fetch_recommendations(self, api_key: str) -> LibraryData:
        k, limit, threshold = self.k, self.limit, self.threshold
        kept_albums, kept_ids = [], set()
        albums_looked = 0
        message = [f"Fetching recommendations. 0/{k} albums found. {albums_looked} albums looked up."]
        stop = loading_animation(message)
        while len(kept_albums) < k:
            message[0] = f"Fetching recommendations. {len(kept_albums)}/{k} albums found. {albums_looked} albums looked up."

            random_tokens = self.explorer._random_artist_genre_tag_generator(1)
            fetched_albums = self._fetch_albums(random_tokens, api_key, limit)
            if not fetched_albums:
                continue
            fetched_albums_gspace_matrix, fetched_albums_tspace_matrix = self._space_matrix(fetched_albums)

            genre_similarity_projections = cosine_similarity(fetched_albums_gspace_matrix, self.taste_gv)
            tag_similarity_projections = cosine_similarity(fetched_albums_tspace_matrix, self.taste_tv)
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
    
    def _fetch_similar_by_mbid(self, mbid: str, api_key: str, limit: int = 10, LASTFM_URL: str = "https://ws.audioscrobbler.com/2.0/") -> list[dict[str, str | float]]:
        params = {
            "method": "artist.getSimilar",
            "mbid": mbid,
            "api_key": api_key,
            "format": "json",
            "limit": limit
        }

        try:
            r = requests.get(LASTFM_URL, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()

            artists = data.get("similarartists", {}).get("artist", [])

            return [
                {
                    "name": a["name"],
                    "mbid": a.get("mbid"),
                    "match": float(a.get("match", 0))
                }
                for a in artists if a.get("name")
            ]

        except requests.RequestException as e:
            logging.warning(f"Error fetching MBID {mbid}: {e}")
            return []
        

    
    def _fetch_similar(self, artist: list[str], api_key:str, limit: int = 100, sleep_time: float = 1.0) -> str | None:
        if not api_key:
            logging.warning(f"No Last.fm API key set. Skipping similar artist fetching.")
            return {}

        results = []
        for artist_name in artist:

            mbid = [a.get("artist_id") for a in self.library if artist_name in a.get("artist", [])][0]
            similar = self._fetch_similar_by_mbid(mbid, api_key, limit)
            results.extend(similar)

            time.sleep(sleep_time)

        random_artist = self.explorer._pick_random_similar(results)

        return random_artist
    
    def _save_file(self, dictionary: dict, write_json=True):
        save_dir = output_path("data")
        os.makedirs(save_dir, exist_ok=True)

        if write_json:
            file_name = f"Last.fm-similar_artists-Dig-for-Fire.json"
            output_file = os.path.join(save_dir, file_name)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(dictionary, f, indent=4, ensure_ascii=False)

        file_name = f"Last.fm-similar_artists-Dig-for-Fire.pkl"
        output_file = os.path.join(save_dir, file_name)
        tmp_file = output_file + ".tmp"

        with open(tmp_file, "wb") as f:
            pickle.dump(dictionary, f)
        os.replace(tmp_file, output_file)
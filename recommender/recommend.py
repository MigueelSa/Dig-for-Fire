import logging
import os, json, random
import numpy as np
from sklearn.decomposition import TruncatedSVD
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

    def __init__(self, library_path: str, email: str, app_name: str = "Dig-for-Fire", app_version: str = "0.1", k: int = 2, limit: int = 1, threshold: float = 0.6):
        self.mb_library                                                                         =   MusicBrainz(app_name, app_version, email)
        self.library                                                                            =   self._load_library(library_path)
        self.roots                                                                              =   self._roots() 
        self.library_embeddings                                                                 =   self._load_embeddings()
        self.taste_gv, self.taste_pv, self.taste_tv                                             =   self._taste_vectors()
        self.app_name, self.app_version, self.email, self.k, self.limit, self.threshold         =   app_name, app_version, email, k, limit, threshold   
        self.used_tokens                                                                        =   set()
        self.excluded_albums                                                                    =   self._excluded_albums()
        self.genre_counts, self.root_counts, self.tag_counts, self.unseen_siblings              =   self._pools()

    def _fetch_recommendations(self) -> LibraryData:
        k, limit, threshold = self.k, self.limit, self.threshold
        kept_albums, kept_ids = [], set()
        with tqdm(total=k, desc="Albums found.", leave = False) as pbar:
            while len(kept_albums) < k:
                random_tokens = self._genre_tag_randomizer(1)
                fetched_albums = self._fetch_albums(random_tokens, limit)
                if not fetched_albums:
                    continue
                fetched_albums_gspace_matrix, fetched_albums_pspace_matrix, fetched_albums_tspace_matrix = self._space_matrix(fetched_albums)
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

    def _album_genres_tags(self, album: AlbumData) -> tuple[dict[str, int], list[str]]:
        """
        Extract genres with minimal distance from leaf genres and other tags from an album.

        Each genre in the returned dictionary includes its minimal distance from the
        album's original genres (0 for direct genres, 1+ for ancestor genres).

        :param self: Instance of the Recommender class.
        :param album: Album data.
        :type album: AlbumData
        :return: A tuple containing:
            - A dictionary mapping genre names to their minimal distance from the album's genres.
            - A list of other tags associated with the album.
        :rtype: tuple[dict[str, int], list[str]]
        """
        genres, tags = album.get("genres", {}), list(album.get("tags", []))
        
        return genres, tags
    

    def _taste_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gv, tv = None, None

        for album in self.library:
            album_genres, album_tags = self._album_embeddings(album)

            if album_genres.all():
                gv_vec = album_genres
                gv = gv_vec if gv is None else gv + gv_vec

            if album_tags.all():
                tv_vec = album_tags
                tv = tv_vec if tv is None else tv + tv_vec

        gv = normalize(gv.reshape(1, -1)) if gv is not None else np.zeros((1, len(next(iter(self.library_embeddings.values())))))
        tv = normalize(tv.reshape(1, -1)) if tv is not None else np.zeros((1, len(next(iter(self.library_embeddings.values())))))

        return gv, tv


    def _pools(self) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, list[str]]]:
        """
        Compute aggregate counts and unseen children for genres and tags in the library.

        This function processes all albums in the library to produce summary data:

        1. Counts of each individual genre across all albums (`genre_counts`).
        2. Counts of each root genre (genres in `self._roots`) across all albums (`root_counts`).
        3. Counts of all tags across all albums (`tag_counts`).
        4. A mapping of genres in the library to their children that do not appear in the library (`unseen_siblings`).

        :param self: Instance of the class containing the library and the MusicBrainz tags data.
        :return: A tuple containing:
            - genre_counts (dict[str, int]): Number of occurrences of each individual genre.
            - root_counts (dict[str, int]): Number of occurrences of each root genre.
            - tag_counts (dict[str, int]): Number of occurrences of each tag across all albums.
            - unseen_siblings (dict[str, list[str]]): Genres mapped to their children that are missing from the library.
        :rtype: tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, list[str]]]
        """

        all_genres, all_roots, all_tags = [], []
        for album in self.library:
            album_genres, album_tags = self._album_genres_tags(album)
            for genre in album_genres.keys():
                if genre in self._roots:
                    all_roots.append(genre)
                else:
                    all_genres.append(genre)
            all_tags.extend(album_tags)
        genre_counts, root_counts, tag_counts = dict(Counter(all_genres)), dict(Counter(all_roots)), dict(Counter(all_tags))

        ancestor_children = self.mb_library.tags._get_ancestors_children()
        unseen_siblings = {}
        for genre in genre_counts.keys():
            children = ancestor_children.get(genre, [])
            unseen = [c for c in children if c not in genre_counts]
            if unseen:
                unseen_siblings[genre] = unseen
        unseen_siblings = set(unseen_siblings)

        return genre_counts, root_counts, tag_counts, unseen_siblings

    def _genre_tag_randomizer(self, k: int, g_probability: float = 0.6, r_probability: float = 0.25) -> list[tuple[str, str]]:
        """
        Randomly select genres, roots, and tags from the library with weighted probabilities.

        Selection rules:
        1. Weighted random pick of non-root genres (`g_probability` chance).
        2. Weighted random pick of a root, then a random unseen child of that root (`r_probability` chance).
        3. Weighted random pick of tags for the remaining probability.

        Duplicates are avoided with `self.used_tokens` and within the current call.

        :param k: Number of tokens to select.
        :param g_probability: Probability of selecting a genre.
        :param r_probability: Probability of selecting a root + unseen child.
        :return: A list of tuples `(token, type)` where type is "genre" or "tag".
        :rtype: list[tuple[str, str]]
        """
        
        genres, roots, tags, unseen_siblings = self.genre_counts, self.root_counts, self.tag_counts, self.unseen_siblings

        g_items, g_weights = [], []
        for genre, weight in genres.items():
            if genre not in self.used_tokens:
                g_items.append(genre)
                g_weights.append(weight)

        r_items, r_weights = [], []
        for root, weight in roots.items():
            if unseen_siblings.get(root) and root not in self.used_tokens:
                r_items.append(root)
                r_weights.append(weight)

        t_items, t_weights = [], []
        for tag, weight in tags.items():
            if tag not in self.used_tokens:
                t_items.append(tag)
                t_weights.append(weight)

        random_tokens, tokens_set = [], set()
        while len(random_tokens) < k:
            selected, r = None, random.random()
            if r < g_probability and g_items:
                selected = random.choices(g_items, weights=g_weights, k=1)[0], "genre"
            elif r < g_probability + r_probability and r_items:
                root = random.choices(r_items, weights=r_weights, k=1)[0]
                selected = random.choice(unseen_siblings.get(root, [])), "genre"
            elif t_items:
                selected = random.choices(t_items, weights=t_weights, k=1)[0], "tag"

            if selected is None:
                available_items = [(x, "genre") for x in g_items + r_items] + [(x, "tag") for x in t_items]
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

    """
    def _albums_parents(self, album: AlbumData) -> list[str]:
        genres, _ = self._album_genres_tags(album)
        parents = set()
        for genre in genres.keys():
            genre_parents = self.mb_library.tags.parents.get(genre) or []
            parents.update(genre_parents)
        return list(parents)
    """
    
    def _coocurrence_matrix(self) -> tuple[np.ndarray, set[str, int]]:
        """
        Compute the co-occurrence matrix for genres, parents, and tags in the library.
        
        :param self: Instance of the Recommender class.
        :param library: Library data.
        :type library: LibraryData
        :return: The co-occurrence matrix as a NumPy ndarray.
        :rtype: ndarray[_AnyShape, dtype[Any]]
        """
        tokens = set()
        for album in self.library:
            genres, parents, tags = self._album_genres_parents_tags(album)
            tokens.update(genres)
            tokens.update(parents)
            tokens.update(tags)
        tokens = sorted(tokens)

        token_index = {token: idx for idx, token in enumerate(tokens)}
        n = len(tokens)
        cooc_matrix = np.zeros((n, n), dtype=int)

        for album in self.library:
            genres, parents, tags = self._album_genres_parents_tags(album)
            album_tokens = list(set(genres + parents + tags))
            for i, token1 in enumerate(album_tokens):
                idx1 = token_index[token1]
                for token2 in album_tokens[i+1:]:
                    idx2 = token_index[token2]
                    cooc_matrix[idx1, idx2] += 1
                    cooc_matrix[idx2, idx1] += 1

        return cooc_matrix, token_index
    
    def _compute_embeddings(self) ->  dict[str, np.ndarray]:
        """
        Compute embeddings for each token in the library.
        
        :param self: Instance of the Recommender class.
        :return: Dictionary mapping each token to its normalized vector.
        """
        
        # The co-occurrence matrix can often be written in block-diagonal form. For each block, we compute eigenvalues and eigenvectors.
        # Each eigenvector represents a direction in genre space, and the projection of all genre vectors onto that eigenvector is related to its eigenvalue.
        # The eigenvector with the smallest eigenvalue (in absolute value) corresponds to the direction along which the projections of the genre vectors are smallest.
        # Therefore, it contributes the least to representing similarities and can be discarded when reducing dimensionality.
        # If we exclude self-occurrences (the diagonal), the eigenvectors capture only co-occurrence with *other* genres.
        # Including the diagonal emphasizes each genre’s own frequency, which can dominate the largest eigenvector. Excluding it highlights relationships between genres.
        coocurrence_matrix, token_index = self._coocurrence_matrix()
        # Here, we will try to keep the same dimension so information is not lost.
        n = coocurrence_matrix.shape[0]
        svd = TruncatedSVD(n_components=n)
        reduced_matrix = svd.fit_transform(coocurrence_matrix)
        reduced_matrix = normalize(reduced_matrix)

        token_embeddings = {token: reduced_matrix[idx] for token, idx in token_index.items()}

        return token_embeddings
    
    def _save_embeddings(self, embeddings: dict[str, np.ndarray], save_dir = output_path("data")) -> None:
        file_path = output_path(save_dir, "embeddings-Dig-for-Fire.npz")

        tokens, vectors = np.array(list(embeddings.keys())), np.stack(list(embeddings.values()))

        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(file_path, tokens=tokens, vectors=vectors)
    
    def _load_embeddings(self, save_dir=output_path("data")) -> dict[str, np.ndarray]:
        file_name = "embeddings-Dig-for-Fire.npz"
        file_path = output_path(save_dir, file_name)
        if not os.path.exists(file_path):
            embeddings = self._compute_embeddings()
            self._save_embeddings(embeddings, save_dir=save_dir)
            return embeddings
        data = np.load(file_path)
        tokens, vectors = data["tokens"], data["vectors"]

        return {token: vectors[i] for i, token in enumerate(tokens)}
    

    def _album_embeddings(self, album: AlbumData) -> tuple[np.ndarray, np.ndarray]:
        any_token = next(iter(self.library_embeddings))
        vector_dim = self.library_embeddings[any_token].shape[0]
        genres, tags = self._album_genres_tags(album)

        zero_array = np.zeros(vector_dim, dtype=float)

        album_genres = np.zeros(vector_dim, dtype=float)
        for genre in genres:
            album_genres += self.library_embeddings.get(genre, zero_array)

        album_tags = np.zeros(vector_dim, dtype=float)
        for tag in tags:
            album_tags += self.library_embeddings.get(tag, zero_array)

        return album_genres, album_tags
    
    def _space_matrix(self, library: LibraryData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gv_list, tv_list = [], [], []
        for album in library:
            g, t = self._album_embeddings(album)
            gv_list.append(g)
            tv_list.append(t)

        G = normalize(np.vstack(gv_list))
        T = normalize(np.vstack(tv_list))

        return G, T
    
    def _roots(self) -> set[str]:
        """
        Return the set of root genres in the library.

        A root genre is defined as a genre that has no direct ancestors
        in the `ancestors` mapping (its ancestor list is empty).

        :return: Set of root genre names.
        :rtype: set[str]
        """
        roots = set(genre for genre, ancestors in self.mb_library.tags.ancestors.items() if not ancestors)
        return roots

    
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
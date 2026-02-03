import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import os, hashlib, json, multiprocessing
from typing import Literal
import networkx as nx
from node2vec import Node2Vec
import tqdm as tqdm

from models.models import AlbumData, LibraryData, MethodType
from utils.paths import output_path
from utils.albums import get_album_genres, get_album_tags
from utils.loading import loading_animation
from tags.tags import Tags

class Embeddings:

    def __init__(self, library: LibraryData, method: MethodType, n: int | None = None, alpha: float = 1, tw: float = 6, rw: float = 6, smooth: bool = True, gamma: float = 0.2, beta: float = 0.1):
        self.tags                       =   Tags()
        self.vocabulary                 =   self._build_vocabulary(library)
        if n is not None and method == "svd":
            self.dimension              =   n
        else:
            self.dimension              =   len(self.vocabulary)
        self.library_embeddings         =   self._load_embeddings(library, method)
        if smooth:
            self.library_embeddings     =   self._smooth(self.library_embeddings, library, gamma=gamma, beta=beta)
        self.alpha, self.tw, self.rw    =   alpha, tw, rw

    def _coocurrence_matrix(self, library: LibraryData, null_diagonal: bool = True) -> tuple[np.ndarray, dict[str, int]]:
        """
        Compute the co-occurrence matrix for genres, parents, and tags in the library.
        
        :param self: Instance of the Embeddings class.
        :param library: Library data.
        :type library: LibraryData
        :return: The co-occurrence matrix as a NumPy ndarray.
        :rtype: ndarray[_AnyShape, dtype[Any]]
        """
        
        tokens = self.vocabulary

        token_index = {token: idx for idx, token in enumerate(tokens)}
        n = len(tokens)
        cooc_matrix = np.zeros((n, n), dtype=int)

        if null_diagonal:
            d = 1
        else:
            d = 0

        for album in library:
            genres, tags = get_album_genres(album).keys(), get_album_tags(album).keys()
            album_tokens = list(genres | tags)
            for i, token1 in enumerate(album_tokens):
                idx1 = token_index[token1]
                for token2 in album_tokens[i+d:]:
                    idx2 = token_index[token2]
                    cooc_matrix[idx1, idx2] += 1
                    cooc_matrix[idx2, idx1] += 1

        return cooc_matrix, token_index
    
    def _compute_pmi_embeddings(self, library: LibraryData, ppmi: bool = False) -> dict[str, np.ndarray]:
        coocurrence_matrix, token_index = self._coocurrence_matrix(library, null_diagonal=False)
        pij = coocurrence_matrix.astype(np.float32)
        total_albums = len(library)
        eps = 1e-18

        pij /= total_albums
        pi, pj = np.diagonal(pij).reshape(-1,1), np.diagonal(pij).reshape(1,-1)

        pmi_matrix = np.log(pij/(pi*pj) + eps)
        if ppmi:
            pmi_matrix = np.maximum(pmi_matrix, 0)
        pmi_matrix = normalize(pmi_matrix)

        return {token: pmi_matrix[idx] for token, idx in token_index.items()}


    
    def _compute_embeddings(self, library: LibraryData, method: MethodType = "ppmi") ->  dict[str, np.ndarray]:
        if method == "cooc":
            cooc_matrix, token_index = self._coocurrence_matrix(library)
            cooc_matrix = normalize(cooc_matrix)
            token_embeddings = {token: cooc_matrix[idx] for token, idx in token_index.items()}
        elif method == "svd":
            token_embeddings = self._compute_svd_embeddings(library)
        elif method == "pmi":
            token_embeddings = self._compute_pmi_embeddings(library)
        elif method == "ppmi":
            token_embeddings = self._compute_pmi_embeddings(library, ppmi=True)
        # elif method == "w2v": # more fit for larger libraries where we can profit from reducing dimension
            # The goal is that the probability of a context given a genre reflects the data.
            # For that, for each genre, two vectors of dimension d are randomly initialized, an input vector v and output vector u.
            # The probabilities are computed by applying a softmax function to these dot products.
            # If the context is around genre in data --> increase dot product vg . uc. If it is not --> decrease dot product vg . uc. This is done in the gradient step.
            # Repeat until vectors converge.

        return token_embeddings
    
    def _compute_svd_embeddings(self, library: LibraryData) ->  dict[str, np.ndarray]:
        """
        Compute svd embeddings for each token in the library.
        
        :param self: Instance of the Embeddings class.
        :return: Dictionary mapping each token to its normalized vector.
        """
        # The co-occurrence matrix can often be written in block-diagonal form. For each block, we compute eigenvalues and eigenvectors.
        # Each eigenvector represents a direction in genre space, and the projection of all genre vectors onto that eigenvector is related to its eigenvalue.
        # The eigenvector with the smallest eigenvalue (in absolute value) corresponds to the direction along which the projections of the genre vectors are smallest.
        # Therefore, it contributes the least to representing similarities and can be discarded when reducing dimensionality.
        # If we exclude self-occurrences (the diagonal), the eigenvectors capture only co-occurrence with *other* genres.
        # Including the diagonal emphasizes each genre’s own frequency, which can dominate the largest eigenvector. Excluding it highlights relationships between genres.
        coocurrence_matrix, token_index = self._coocurrence_matrix(library)
        # Here, we will try to keep the same dimension so information is not lost. This is the same as using the co-ocurrence matrix.
        svd = TruncatedSVD(n_components=self.dimension)
        reduced_matrix = svd.fit_transform(coocurrence_matrix)
        reduced_matrix = normalize(reduced_matrix)

        token_embeddings = {token: reduced_matrix[idx] for token, idx in token_index.items()}

        return token_embeddings
    
    def _save_embeddings(self, embeddings: dict[str, np.ndarray], library: LibraryData, method: MethodType, save_dir = output_path("data")) -> None:
        lib_hash = self._vocabulary_hash(library, method)
        file_name = f"embeddings-{method}-{lib_hash}-{self.dimension}-Dig-for-Fire.npz"
        file_path = output_path(save_dir, file_name)

        tokens, vectors = np.array(list(embeddings.keys())), np.stack(list(embeddings.values()))

        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(file_path, tokens=tokens, vectors=vectors)

    def _load_embeddings(self, library: LibraryData, method: MethodType, save_dir=output_path("data")) -> dict[str, np.ndarray]:
        lib_hash = self._vocabulary_hash(library, method)
        file_name = f"embeddings-{method}-{lib_hash}-{self.dimension}-Dig-for-Fire.npz"
        file_path = output_path(save_dir, file_name)
        if not os.path.exists(file_path):
            embeddings = self._compute_embeddings(library, method=method)
            self._save_embeddings(embeddings, library, method, save_dir=save_dir)
            return embeddings
        data = np.load(file_path)
        tokens, vectors = data["tokens"], data["vectors"]

        return {token: vectors[i] for i, token in enumerate(tokens)}
    
    def get_album_embeddings(self, album: AlbumData, token_type: Literal["genres", "tags"]) -> np.ndarray:
        alpha = self.alpha

        if token_type == "genres":
            tokens = get_album_genres(album)
        elif token_type == "tags":
            tokens = get_album_tags(album)
            alpha = alpha * self.tw

        zero_array = np.zeros(self.dimension, dtype=float)
        album_tokens = np.zeros(self.dimension, dtype=float)
        roots = self.tags.roots
        for token, distance in tokens.items():
            if token not in roots:
                album_tokens += self.library_embeddings.get(token, zero_array) * np.exp(-alpha*distance)
            else:
                album_tokens += self.library_embeddings.get(token, zero_array) * np.exp(-alpha*distance*self.rw)

        if np.linalg.norm(album_tokens) > 0:
            album_tokens = album_tokens / np.linalg.norm(album_tokens)

        return album_tokens
    
    def _build_vocabulary(self, library: LibraryData) -> set[str]:
        vocabulary = set()
        for album in library:
            album_genres, album_tags = get_album_genres(album).keys(), get_album_tags(album).keys()
            vocabulary.update(album_genres | album_tags)

        return sorted(vocabulary)
    
    def _vocabulary_hash(self, library: LibraryData, method: MethodType) -> str:
        library_data = [
            {
                "genres": list(get_album_genres(a).keys()),
                "tags": list(get_album_tags(a).keys())
            }
            
            for a in library
        ]

        payload = json.dumps({"library": library_data, "method": method}, sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()
    
    def _initialize_genre_embeddings(self, embeddings: dict[str, np.ndarray], gamma: float = 0.2) -> dict[str, np.ndarray]:
        genre_vectors = {g: v.copy() for g, v in embeddings.items()}

        new_vectors = {}
        for genre in self.tags.canonical_genres:
            if genre not in self.vocabulary:
                continue
            ancestors = self.tags.ancestors.get(genre, [])
            ancestor_vecs = np.sum([genre_vectors[ancestor] for ancestor in ancestors], axis=0)
            if ancestors:
                vec = (1-gamma) * genre_vectors[genre] + gamma * ancestor_vecs
                new_vectors[genre] = normalize(vec.reshape(1,-1))[0]
            else:
                new_vectors[genre] = genre_vectors[genre]

        return new_vectors
    
    def _smooth_by_coocurrence(self, embeddings: dict[str, np.ndarray], library: LibraryData, gamma: float = 0.2, beta: float = 0.1) -> dict[str, np.ndarray]:
        genre_vectors = self._initialize_genre_embeddings(embeddings, gamma = gamma)
        cooc_matrix, token_index = self._coocurrence_matrix(library)
        cooc_weights = cooc_matrix.astype(float)
        cooc_weights /= cooc_weights.sum(axis=1, keepdims=True) + 1e-18
        
        smoothed_vectors = {}
        for genre in genre_vectors:
            if genre not in token_index:
                smoothed_vectors[genre] = genre_vectors[genre]
                continue
            idx = token_index[genre]
            neighbor_mean = np.sum([cooc_weights[idx, j] * genre_vectors[token] for token, j in token_index.items() if token in self.tags.canonical_genres], axis=0)
                    
            if np.linalg.norm(neighbor_mean) > 0:
                vec = (1 - beta) * genre_vectors[genre] + beta * neighbor_mean
                smoothed_vectors[genre] = normalize(vec.reshape(1,-1))[0]
            else:
                smoothed_vectors[genre] = genre_vectors[genre]

        return smoothed_vectors
    
    """
    
    def _initialize_genre_embeddings(self, embeddings: dict[str, np.ndarray], gamma: float = 0.2) -> dict[str, np.ndarray]:
        genre_vectors = {g: v.copy() for g, v in embeddings.items()}

        new_vectors = {}
        for genre in self.tags.canonical_genres:
            ancestor_vecs = np.zeros(self.dimension)
            distance_dict, _ = self.tags.genres_tags([genre])
            ancestors = self.tags.ancestors.get(genre, [])
            weights = 0
            for ancestor in ancestors:
                distance = distance_dict[ancestor]
                weight = np.exp(-self.alpha * distance * (self.rw if ancestor in self.tags.roots else 1))
                ancestor_vecs += genre_vectors[ancestor]*weight
                weights += weight
            if weights > 0:
                ancestor_vecs /= weights
            if ancestors:
                new_vectors[genre] = normalize((1-gamma) * genre_vectors[genre] + gamma * ancestor_vecs)
            else:
                new_vectors[genre] = genre_vectors[genre]

        return new_vectors
    
    def _smooth_by_coocurrence(self, embeddings: dict[str, np.ndarray], library: LibraryData, gamma: float = 0.2, beta: float = 0.1) -> dict[str, np.ndarray]:
        genre_vectors = self._initialize_genre_embeddings(embeddings, gamma = gamma)
        cooc_matrix, token_index = self._coocurrence_matrix(library)
        cooc_weights = cooc_matrix.astype(float)
        cooc_weights /= cooc_weights.sum(axis=1, keepdims=True) + 1e-18
        
        smoothed_vectors = {}
        for genre in genre_vectors:
            distance_dict, _ = self.tags.genres_tags([genre])
            if genre not in token_index:
                smoothed_vectors[genre] = genre_vectors[genre]
                continue
            idx = token_index[genre]
            neighbor_mean = np.zeros(self.dimension)
            for token, j in token_index.items():
                distance = distance_dict.get(token, 1.0)
                weight = np.exp(-self.alpha * distance * (self.rw if token in self.tags.roots else 1))
                neighbor_mean += cooc_weights[idx, j] * weight * genre_vectors[token]
                    
            if np.linalg.norm(neighbor_mean) > 0:
                smoothed_vectors[genre] = normalize((1 - beta) * genre_vectors[genre] + beta * neighbor_mean)
            else:
                smoothed_vectors[genre] = genre_vectors[genre]

        return smoothed_vectors

        """

    def _smooth_by_node2vec(self, embeddings: dict[str, np.ndarray], library: LibraryData,gamma: float = 0.2,
        n2v_dim: int | None = None,
        n2v_walk_length: int = 30,
        n2v_num_walks: int = 100,
        n2v_window: int = 10,
        n2v_p: float = 1,
        n2v_q: float = 1,
        n2v_weight: float = 0.5) -> dict[str, np.ndarray]:
        
        G = nx.Graph()
        for genre in self.tags.canonical_genres:
            G.add_node(genre)
            for ancestor in self.tags.ancestors.get(genre, []):
                G.add_edge(genre, ancestor, weight=1.0)

        cooc_matrix, token_index = self._coocurrence_matrix(library)
        for g1, idx1 in token_index.items():
            for g2, idx2 in token_index.items():
                if g1 != g2 and cooc_matrix[idx1, idx2] > 0:
                    weight = cooc_matrix[idx1, idx2]
                    if G.has_edge(g1, g2):
                        G[g1][g2]['weight'] += weight
                    else:
                        G.add_edge(g1, g2, weight=weight)

        if n2v_dim is None:
            n2v_dim = self.dimension

        stop = loading_animation("Smoothing embeddings with Node2Vec...")
        workers = max(1, multiprocessing.cpu_count() - 1)
        n2v_walk_length = min(n2v_walk_length, len(G.nodes) // 2)
        node2vec = Node2Vec(G, dimensions=n2v_dim, walk_length=n2v_walk_length, num_walks=n2v_num_walks, p=n2v_p, q=n2v_q, weight_key = "weight",workers=workers, seed = 1, quiet = True)
        n2v_model = node2vec.fit(window=n2v_window, min_count=1, seed = 1)
        stop()

        genre_vectors = self._initialize_genre_embeddings(embeddings, gamma=gamma)
        for genre in genre_vectors:
            if genre in n2v_model.wv:
                n2v_vec = n2v_model.wv[genre]
                merged_vec = (1 - n2v_weight) * genre_vectors[genre] + n2v_weight * n2v_vec
            else:
                merged_vec = genre_vectors[genre]        
            genre_vectors[genre] = normalize(merged_vec.reshape(1,-1))[0]

        return genre_vectors
    
    def _smooth(self, embeddings: dict[str, np.ndarray], library: LibraryData, smooth_type: Literal["cooc", "n2v"] = "n2v", gamma: float = 0.2, beta: float = 0.1) -> dict[str, np.ndarray]:
        if smooth_type == "cooc":
            return self._smooth_by_coocurrence(embeddings, library, gamma=gamma, beta=beta)
        elif smooth_type == "n2v":
            return self._smooth_by_node2vec(embeddings, library, gamma=gamma)

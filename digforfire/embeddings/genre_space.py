from digforfire.embeddings.embeddings import Embeddings
from digforfire.models.models import AlbumData, LibraryData, EmbeddingData, MethodType
from digforfire.utils.albums import get_album_genres
from digforfire.utils.paths import output_path
from digforfire.utils.loading import loading_animation

import multiprocessing, hashlib, pickle, os
import networkx as nx
from node2vec import Node2Vec
from gensim.models import KeyedVectors
from typing import Literal
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from scipy.sparse import coo_matrix

class GenreSpace(Embeddings):
    def __init__(self, library: LibraryData, method: MethodType = "pmi"):
        super().__init__(library, token_type="genre", method=method)

        self.vocabulary: list[str]                      =   self._build_vocabulary()
        self.dimension: int                             =   len(self.vocabulary)
        self.library_embeddings: EmbeddingData          =   self._load_embeddings(self.dimension)

    def _cooccurrence_matrix(self, null_diagonal: bool = True) -> tuple[coo_matrix, dict[str, int]]:
        
        tokens = self.vocabulary

        token_index = {token: idx for idx, token in enumerate(tokens)}

        if null_diagonal:
            d = 1
        else:
            d = 0

        rows, cols, data = [], [], []
        for album in self.library:
            album_tokens = [token_index[t] for t in get_album_genres(album).keys()]
            for i, idx1 in enumerate(album_tokens):
                for idx2 in album_tokens[i+d:]:
                    rows.append(idx1)
                    cols.append(idx2)
                    data.append(1)
                    if idx1 != idx2:
                        rows.append(idx2)
                        cols.append(idx1)
                        data.append(1)

        n = len(token_index)
        cooc_matrix = coo_matrix((data, (rows, cols)), shape=(n, n))

        return cooc_matrix, token_index
    
    def _compute_pmi_embeddings(self, ppmi: bool = False) -> EmbeddingData:
        cooc_matrix, token_index = self._cooccurrence_matrix()
        pmi = self._sparse_pmi(cooc_matrix, ppmi=ppmi)
        return {token: pmi.getrow(idx).toarray().ravel() for token, idx in token_index.items()}


    
    def _compute_embeddings(self, **kwargs) ->  EmbeddingData:
        if self.method == "cooc":
            cooc_matrix, token_index = self._cooccurrence_matrix()
            cooc_matrix = normalize(cooc_matrix)
            token_embeddings = {token: cooc_matrix[idx] for token, idx in token_index.items()}
        elif self.method == "svd":
            token_embeddings = self._compute_svd_embeddings(**kwargs)
        elif self.method == "pmi":
            token_embeddings = self._compute_pmi_embeddings()
        elif self.method == "ppmi":
            token_embeddings = self._compute_pmi_embeddings(ppmi=True)
        elif self.method == "tag_clusters":
            raise ValueError("Tag clusters method is not applicable for genre embeddings.")
        else:
            raise ValueError(f"Unknown method: {self.method}")
        # elif method == "w2v": # more fit for larger libraries where we can profit from reducing genre_embedding_dimension
            # The goal is that the probability of a context given a genre reflects the data.
            # For that, for each genre, two vectors of dimension d are randomly initialized, an input vector v and output vector u.
            # The probabilities are computed by applying a softmax function to these dot products.
            # If the context is around genre in data --> increase dot product vg . uc. If it is not --> decrease dot product vg . uc. This is done in the gradient step.
            # Repeat until vectors converge.

        return token_embeddings
    
    def _compute_svd_embeddings(self, **kwargs) ->  EmbeddingData:
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
        cooccurrence_matrix, token_index = self._cooccurrence_matrix()
        # Here, we will try to keep the same dimension so information is not lost. This is the same as using the co-occurrence matrix.
        n_reduced = kwargs.get("n_reduced", len(self.vocabulary))
        svd = TruncatedSVD(n_components=n_reduced)
        reduced_matrix = svd.fit_transform(cooccurrence_matrix)
        reduced_matrix = normalize(reduced_matrix)

        token_embeddings = {token: reduced_matrix[idx] for token, idx in token_index.items()}

        return token_embeddings
    
    def _initialize_genre_embeddings(self, embeddings: EmbeddingData, gamma: float = 0.2) -> EmbeddingData:
        genre_vectors = {g: v.copy() for g, v in embeddings.items()}

        new_vectors = {}
        for genre in self.vocabulary:
            ancestors = self.tags.ancestors.get(genre, [])
            ancestor_vecs = np.sum([genre_vectors[ancestor] for ancestor in ancestors], axis=0)
            if ancestors:
                vec = (1-gamma) * genre_vectors[genre] + gamma * ancestor_vecs
                new_vectors[genre] = normalize(vec.reshape(1,-1))[0]
            else:
                new_vectors[genre] = genre_vectors[genre]

        return new_vectors
    
    def _smooth_by_cooccurrence(self, embeddings: EmbeddingData, gamma: float = 0.2, beta: float = 0.1) -> EmbeddingData:
        genre_vectors = self._initialize_genre_embeddings(embeddings, gamma = gamma)
        cooc_matrix, token_index = self._cooccurrence_matrix()
        cooc_weights = cooc_matrix.astype(float)
        row_sums = np.array(cooc_weights.sum(axis=1)).flatten()
        cooc_weights = cooc_weights.multiply(1 / (row_sums[:, np.newaxis] + 1e-18)).toarray()
        
        smoothed_vectors = {}
        genre_list = list(token_index.keys())
        genre_matrix = np.vstack([genre_vectors[token] for token in genre_list])
        for genre in genre_vectors:
            if genre not in token_index:
                smoothed_vectors[genre] = genre_vectors[genre]
                continue
            idx = token_index[genre]
            neighbor_mean = cooc_weights[idx, :].dot(genre_matrix)
                    
            if np.linalg.norm(neighbor_mean) > 0:
                vec = (1 - beta) * genre_vectors[genre] + beta * neighbor_mean
                smoothed_vectors[genre] = normalize(vec.reshape(1,-1))[0]
            else:
                smoothed_vectors[genre] = genre_vectors[genre]

        return smoothed_vectors

    def _smooth_by_node2vec(self, embeddings: EmbeddingData, gamma: float = 0.2, 
                            n2v_dim: int | None = None, n2v_walk_length: int = 30, n2v_num_walks: int = 100, 
                            n2v_window: int = 10, n2v_p: float = 1, n2v_q: float = 1, n2v_weight: float = 0.5) -> EmbeddingData:
        
        G = nx.Graph()
        for genre in self.tags.canonical_genres:
            G.add_node(genre)
            for ancestor in self.tags.ancestors.get(genre, []):
                G.add_edge(genre, ancestor, weight=1.0)

        cooc_matrix, _ = self._cooccurrence_matrix()
        cooc_matrix_coo = cooc_matrix.tocoo()
        for i, j, v in zip(cooc_matrix_coo.row, cooc_matrix_coo.col, cooc_matrix_coo.data):
            g1, g2 = self.vocabulary[i], self.vocabulary[j]
            if g1 != g2:
                if G.has_edge(g1, g2):
                    G[g1][g2]['weight'] += v
                else:
                    G.add_edge(g1, g2, weight=v)

        if n2v_dim is None:
            n2v_dim = len(self.vocabulary)

        graph_model_filename = self._graph_hash_filename(G, n2v_dim, n2v_walk_length, n2v_num_walks, n2v_window, n2v_p, n2v_q, seed=1)
        graph_model_path = output_path("data", graph_model_filename)
        if os.path.exists(graph_model_path):
            n2v_model = KeyedVectors.load(graph_model_path)
        else:
            stop = loading_animation("Smoothing embeddings with Node2Vec...")
            workers = max(1, multiprocessing.cpu_count() - 1)
            n2v_walk_length = min(n2v_walk_length, len(G.nodes) // 2)
            node2vec = Node2Vec(G, dimensions=n2v_dim, walk_length=n2v_walk_length, num_walks=n2v_num_walks, p=n2v_p, q=n2v_q, weight_key = "weight",workers=workers, seed = 1, quiet = True)
            n2v_model = node2vec.fit(window=n2v_window, min_count=1, seed = 1)
            n2v_model.wv.save(graph_model_path)
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
    
    def _smooth(self, embeddings: EmbeddingData, **kwargs) -> EmbeddingData:
        smooth_type: Literal["cooc", "n2v"] = kwargs.get("smooth_type", "n2v")
        gamma: float = kwargs.get("gamma", 0.2)
        beta: float = kwargs.get("beta", 0.1)

        if smooth_type == "cooc":
            return self._smooth_by_cooccurrence(embeddings, gamma=gamma, beta=beta)
        elif smooth_type == "n2v":
            return self._smooth_by_node2vec(embeddings, gamma=gamma)
        else:
            raise ValueError(f"Unknown smooth_type: {smooth_type}")
    
    def _build_vocabulary(self) -> list[str]:
        genres = set()
        for album in self.library:
            album_genres = get_album_genres(album).keys()
            genres.update(album_genres)
        vocabulary = sorted(list(genres))
        return vocabulary
    
    def get_album_embeddings(self, album: AlbumData, **kwargs) -> np.ndarray:
        alpha, rw = kwargs.get("alpha", 1.0), kwargs.get("rw", 6.0)
        tokens = get_album_genres(album)
        dimension = len(next(iter(self.library_embeddings.values())))

        zero_array = np.zeros(dimension, dtype=float)
        album_tokens = np.zeros(dimension, dtype=float)
        roots = self.tags.roots
        for token, distance in tokens.items():
            if token not in roots:
                album_tokens += self.library_embeddings.get(token, zero_array) * np.exp(-alpha*distance)
            else:
                album_tokens += self.library_embeddings.get(token, zero_array) * np.exp(-alpha*distance*rw)

        if np.linalg.norm(album_tokens) > 0:
            album_tokens = album_tokens / np.linalg.norm(album_tokens)

        return album_tokens
    
    def _graph_hash_filename(self, G, n2v_dim: int, n2v_walk_length=30, n2v_num_walks=100, n2v_window=10, n2v_p=1, n2v_q=1, seed=1) -> str:
        graph_data = [(u, v, G[u][v]["weight"]) for u, v in sorted(G.edges())]
        graph_bytes = pickle.dumps((graph_data, n2v_dim, n2v_walk_length, n2v_num_walks, n2v_window, n2v_p, n2v_q, seed))
        graph_hash = hashlib.md5(graph_bytes).hexdigest()
        return f"node2vec_{graph_hash}.model"
    
    def _sparse_pmi(self, cooccurrence_matrix: coo_matrix, ppmi: bool = False) -> coo_matrix:
        cooccurrence_matrix = cooccurrence_matrix.astype(np.float32)

        total = cooccurrence_matrix.data.sum()
        pij = cooccurrence_matrix.data / total

        pi = cooccurrence_matrix.diagonal().astype(np.float32) / total
        pi[pi == 0] = 1e-18
        pj = pi.copy()

        log_pij = np.log(pij / (pi[cooccurrence_matrix.row] * pj[cooccurrence_matrix.col]) + 1e-18)

        if ppmi:
            log_pij = np.maximum(log_pij, 0)

        pmi = coo_matrix((log_pij, (cooccurrence_matrix.row, cooccurrence_matrix.col)), shape=cooccurrence_matrix.shape)
        pmi = normalize(pmi, axis=1)

        return pmi

        

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import os
from typing import Literal

from models.models import AlbumData, LibraryData
from utils.paths import output_path
from utils.albums import get_album_genres, get_album_tags
from tags.tags import Tags

class Embeddings:

    def __init__(self, library: LibraryData):
        self.library_embeddings     =   self._load_embeddings(library)
        self.tags                   =   Tags()

    def _coocurrence_matrix(self, library: LibraryData) -> tuple[np.ndarray, dict[str, int]]:
        """
        Compute the co-occurrence matrix for genres, parents, and tags in the library.
        
        :param self: Instance of the Embeddings class.
        :param library: Library data.
        :type library: LibraryData
        :return: The co-occurrence matrix as a NumPy ndarray.
        :rtype: ndarray[_AnyShape, dtype[Any]]
        """
        tokens = set()
        for album in library:
            genres, tags = get_album_genres(album), get_album_tags(album)
            tokens.update(genres)
            tokens.update(tags)
        tokens = sorted(tokens)

        token_index = {token: idx for idx, token in enumerate(tokens)}
        n = len(tokens)
        cooc_matrix = np.zeros((n, n), dtype=int)

        for album in library:
            genres, tags = get_album_genres(album), get_album_tags(album)
            album_tokens = list(set(genres + tags))
            for i, token1 in enumerate(album_tokens):
                idx1 = token_index[token1]
                for token2 in album_tokens[i+1:]:
                    idx2 = token_index[token2]
                    cooc_matrix[idx1, idx2] += 1
                    cooc_matrix[idx2, idx1] += 1

        return cooc_matrix, token_index
    
    def _compute_embeddings(self, library: LibraryData, method: Literal["cooc", "svd"] = "cooc") ->  dict[str, np.ndarray]:
        if method == "cooc":
            cooc_matrix, token_index = self._coocurrence_matrix(library)
            cooc_matrix = normalize(cooc_matrix)
            token_embeddings = {token: cooc_matrix[idx] for token, idx in token_index.items()}
        elif method == "svd":
            token_embeddings = self._compute_svd_embeddings(library)
        # elif method == "w2v": # more fit for larger libraries where we can profit from reducing dimension
            # The goal is that the probability of a context given a genre reflects the data.
            # For that, for each genre, two vectors of dimension d are randomly initialized, an input vector v and output vector u.
            # The probabilities are computed by applying a softmax function to these dot products.
            # If the context is around genre in data --> increase dot product vg . uc. If it is not --> decrease dot product vg . uc. This is done in the gradient step.
            # Repeat until vectors converge.

        return token_embeddings
    
    def _compute_svd_embeddings(self, library: LibraryData, n: int | None = None) ->  dict[str, np.ndarray]:
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
        if n is None or n > coocurrence_matrix.shape[0]:
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

    def _load_embeddings(self, library: LibraryData, save_dir=output_path("data")) -> dict[str, np.ndarray]:
        file_name = "embeddings-Dig-for-Fire.npz"
        file_path = output_path(save_dir, file_name)
        if not os.path.exists(file_path):
            embeddings = self._compute_embeddings(library)
            self._save_embeddings(embeddings, save_dir=save_dir)
            return embeddings
        data = np.load(file_path)
        tokens, vectors = data["tokens"], data["vectors"]

        return {token: vectors[i] for i, token in enumerate(tokens)}
    
    def get_album_embeddings(self, album: AlbumData, token_type: Literal["genres", "tags"], vector_dim: int | None = None, alpha: float = 2, tw = 6, tr = 6) -> np.ndarray:
        if vector_dim is None:
            any_token = next(iter(self.library_embeddings))
            vector_dim = self.library_embeddings[any_token].shape[0]

        if token_type == "genres":
            tokens = get_album_genres(album)
        elif token_type == "tags":
            tokens = get_album_tags(album)
            alpha = alpha * tw

        zero_array = np.zeros(vector_dim, dtype=float)
        album_tokens = np.zeros(vector_dim, dtype=float)
        roots = self.tags.roots
        for token, distance in tokens.items():
            if token not in roots:
                album_tokens += self.library_embeddings.get(token, zero_array) * np.exp(-alpha*distance)

        if np.all(album_tokens == 0):
            for token, distance in tokens.items():
                album_tokens += self.library_embeddings.get(token, zero_array) * np.exp(-alpha*distance*tr)

        if np.linalg.norm(album_tokens) > 0:
            album_tokens = album_tokens / np.linalg.norm(album_tokens)

        return album_tokens

    

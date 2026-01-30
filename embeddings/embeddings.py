import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import os, hashlib, json
from typing import Literal

from models.models import AlbumData, LibraryData
from utils.paths import output_path
from utils.albums import get_album_genres, get_album_tags
from tags.tags import Tags

class Embeddings:

    def __init__(self, library: LibraryData, method: Literal["cooc", "svd"], n: int | None = None, alpha: float = 2, tw = 6, tr = 6):
        self.tags                       =   Tags()
        if n is not None and method != "cooc":
            self.dimension              =   n
        else:
            self.dimension              =   len(self._build_vocabulary(library))
        self.library_embeddings         =   self._load_embeddings(library, method)
        self.alpha, self.tw, self.tr    =   alpha, tw, tr

    def _coocurrence_matrix(self, library: LibraryData) -> tuple[np.ndarray, dict[str, int]]:
        """
        Compute the co-occurrence matrix for genres, parents, and tags in the library.
        
        :param self: Instance of the Embeddings class.
        :param library: Library data.
        :type library: LibraryData
        :return: The co-occurrence matrix as a NumPy ndarray.
        :rtype: ndarray[_AnyShape, dtype[Any]]
        """
        
        tokens = self._build_vocabulary(library)

        token_index = {token: idx for idx, token in enumerate(tokens)}
        n = len(tokens)
        cooc_matrix = np.zeros((n, n), dtype=int)

        for album in library:
            genres, tags = get_album_genres(album).keys(), get_album_tags(album).keys()
            album_tokens = list(genres | tags)
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
    
    def _save_embeddings(self, embeddings: dict[str, np.ndarray], library: LibraryData, method: Literal["cooc", "svd"], save_dir = output_path("data")) -> None:
        lib_hash = self._vocabulary_hash(library, method)
        file_name = f"embeddings-{method}-{lib_hash}-{self.dimension}-Dig-for-Fire.npz"
        file_path = output_path(save_dir, file_name)

        tokens, vectors = np.array(list(embeddings.keys())), np.stack(list(embeddings.values()))

        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(file_path, tokens=tokens, vectors=vectors)

    def _load_embeddings(self, library: LibraryData, method: Literal["cooc", "svd"], save_dir=output_path("data")) -> dict[str, np.ndarray]:
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

        if np.all(album_tokens == 0):
            for token, distance in tokens.items():
                album_tokens += self.library_embeddings.get(token, zero_array) * np.exp(-alpha*distance*self.tr)

        if np.linalg.norm(album_tokens) > 0:
            album_tokens = album_tokens / np.linalg.norm(album_tokens)

        return album_tokens
    
    def _build_vocabulary(self, library: LibraryData) -> set[str]:
        vocabulary = set()
        for album in library:
            album_genres, album_tags = get_album_genres(album).keys(), get_album_tags(album).keys()
            vocabulary.update(album_genres | album_tags)

        return sorted(vocabulary)
    
    def _vocabulary_hash(self, library: LibraryData, method: Literal["cooc", "svd"]) -> str:
        library_data = [
            {
                "genres": list(get_album_genres(a).keys()),
                "tags": list(get_album_tags(a).keys())
            }
            
            for a in library
        ]

        payload = json.dumps({"library": library_data, "method": method}, sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()
    



    

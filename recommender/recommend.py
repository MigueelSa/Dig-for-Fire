import os, json
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

class Recommender:
    def __init__(self, library_path: str):
        self.library                =   self._load_library(library_path)
        self.library_space_matrix   =   self._library_space_matrix()
        #self.similarity_matrix      =   cosine_similarity(self.library_space_matrix)
        self.taste_vector           =   self._taste_vector()
        self.similarity_matrix      =   cosine_similarity(self.library_space_matrix, self.taste_vector)


    def recommend(self, library: list[dict], k: int = 5) -> list[dict]:
        """
        Input: your known albums
        Output: k new albums (not in library)
        """
        return

    def _load_library(self, library_path: str) -> dict[str, any]:
        with open(library_path, "r") as file:
            library = json.load(file)
        return library

    def _album_to_tokens(self, album: dict[str, any]) -> list[str]:
        tokens = []
        tokens.extend(album.get("tags", []))
        if "decade" in album:
            tokens.append(album["decade"])
        return tokens

    def _album_tags(self) -> list[str]:
        docs = [" ".join(self._album_to_tokens(a)) for a in self.library]
        return docs

    def _library_space_matrix(self) -> csr_matrix:
        docs = self._album_tags()
        vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"[^ ]+")
        X = vectorizer.fit_transform(docs)
        return X

    def _taste_vector(self) -> np.ndarray:
        taste = self.library_space_matrix.sum(axis=0)
        taste = np.asarray(taste)
        taste = normalize(taste)
        # now this should be true taste.shape == (1, V)
        return taste

if __name__ == "__main__":
    import argparse, time
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--library_path', type=str, required=True, help="Path of the local library.")
    args = parser.parse_args()

    library = os.path.abspath(args.library_path)

    rec = Recommender(library)
    S = rec.similarity_matrix

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Script ran in {elapsed:.2f} seconds")

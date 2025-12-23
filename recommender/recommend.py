import os, json
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, library_path: list[dict]):
        self.library                =   self._load_library(library_path)
        self.library_space_matrix   =   self._library_space_matrix()

    def recommend(library: list[dict], k: int = 5) -> list[dict]:
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

if __name__ == "__main__":
    import argparse, time
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--library_path', type=str, required=True, help="Path of the local library.")
    args = parser.parse_args()

    library = os.path.abspath(args.library_path)

    rec = Recommender(library)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Script ran in {elapsed:.2f} seconds")

import logging, time
import os, json
from libraries.libraries import MusicBrainz

from models.models import LibraryData, MethodType
from recommender.fetcher import Fetcher
from utils.paths import output_path
from embeddings.genre_space import GenreSpace
from embeddings.tag_space import TagSpace

class Recommender:

    def __init__(self, library_path: str, email: str, app_name: str = "Dig-for-Fire", app_version: str = "0.1", k: int = 2, limit: int = 10, threshold: float = 0.6, 
                 method: MethodType = "pmi", n_clusters: int = 50):
        self.mb_library: MusicBrainz                                                            =   MusicBrainz(app_name, app_version, email)
        self.library: LibraryData                                                               =   self.mb_library._load_library(library_path)
        self.roots: set[str]                                                                    =   self.mb_library.tags.roots
        self.genre_embeddings: GenreSpace                                                       =   GenreSpace(self.library, method=method)
        self.tag_embeddings: TagSpace                                                           =   TagSpace(self.library, n_clusters=n_clusters)
        self.recommendation_history: LibraryData                                                =   self._load_recommendations()
        self.fetcher: Fetcher                                                                   =   Fetcher(self.library, self.mb_library, self.genre_embeddings, self.tag_embeddings, 
                                                                                                            self.recommendation_history, k, limit, threshold)
    
    def recommend(self) -> str:
        top_albums = self.fetcher._fetch_recommendations()
        self._save_recommendations(top_albums)
        output = ""
        for ialbum,  album in enumerate(top_albums):
            output += f"{ialbum+1}. {album['title']} by {', '.join(album['artist'])}. Genres: {', '.join(album['genres'])}. Tags: {', '.join(album['tags'])}. Similarity score: {album['score']}.\n"
        return output
    
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
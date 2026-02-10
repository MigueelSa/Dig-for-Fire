from models.models import AlbumData, LibraryData, EmbeddingData
from embeddings.embeddings import Embeddings
from utils.albums import get_album_tags
from utils.loading import loading_animation

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import logging


logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from transformers.utils.logging import disable_progress_bar
disable_progress_bar()


class TagSpace(Embeddings):
    def __init__(self, library: LibraryData, n_clusters: int = 50):
        super().__init__(library, token_type="tag", method="tag_clusters")

        stop = loading_animation("Initializing SentenceTransformer model...")
        self.model                              =   SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        stop()
        
        self.library_mood_tag_embeddings        =   self._library_mood_tag_embeddings(n_clusters=n_clusters)
        self.vocabulary                         =   self._build_vocabulary()
        self.dimension: int                     =   len(self.vocabulary)
        self.library_embeddings: EmbeddingData  =   self._load_embeddings(self.dimension)


    def _compute_tag_clusters(self, tags:list[str], n_clusters: int) -> tuple[EmbeddingData, dict[int, str]]:
        # turns tags into embeddings
        embeddings = self.model.encode(tags, normalize_embeddings=True)

        # groups tags into clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=1)
        labels = kmeans.fit_predict(embeddings)

        cluster_vectors, cluster_representative = {}, {}
        for i in range(n_clusters):
            # goes over all tags in cluster i
            cluster_indices = np.where(labels == i)[0]
            # the vector of the cluster is the mean of its vectors
            cluster_vec = embeddings[cluster_indices].mean(axis=0)
            for idx in cluster_indices:
                cluster_vectors[tags[idx]] = cluster_vec
            # compute distances to the cluster vector
            distances = np.linalg.norm(embeddings[cluster_indices] - cluster_vec, axis=1)
            # select the tag closest to the cluster vector as representative
            cluster_representative[i] = tags[cluster_indices[np.argmin(distances)]]
        
        return cluster_vectors, cluster_representative
    
    def _library_mood_tag_embeddings(self, n_clusters: int = 50) -> EmbeddingData:
        all_tags = set()
        for album in self.library:
            album_tags = get_album_tags(album).keys()
            all_tags.update(album_tags)
        all_tags = sorted(all_tags)

        tag_embeddings, _ = self._compute_tag_clusters(all_tags, n_clusters=n_clusters)

        return tag_embeddings
    
    def _compute_embeddings(self, **kwargs) -> EmbeddingData:
        return self._library_mood_tag_embeddings(n_clusters=kwargs.get("n_clusters", 50))
    
    def _build_vocabulary(self) -> list[str]:
        return sorted(self.library_mood_tag_embeddings.keys())
    
    def get_album_embeddings(self, album: AlbumData, **kwargs) -> np.ndarray:
        tokens = get_album_tags(album).keys()
        vectors = [self.library_mood_tag_embeddings[token] for token in tokens if token in self.library_mood_tag_embeddings]
        if len(vectors) == 0:
            return np.zeros(next(iter(self.library_mood_tag_embeddings.values())).shape)
        album_embedding = normalize(np.sum(vectors, axis=0).reshape(1, -1))[0]
        return album_embedding
    
    def _smooth(self, embeddings: EmbeddingData, **kwargs) -> EmbeddingData:
        # no smoothing for tag space
        return embeddings
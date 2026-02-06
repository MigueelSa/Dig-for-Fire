import numpy as np
import os, json, hashlib
from abc import ABC, abstractmethod

from models.models import AlbumData, LibraryData, EmbeddingData, MethodType, TokenType
from utils.paths import output_path
from utils.albums import get_album_genres, get_album_tags
from tags.tags import Tags

class Embeddings(ABC):

    def __init__(self, library: LibraryData):
        self.tags                           =   Tags()
        self.library: LibraryData           =   library
        self.token_type: TokenType | None   =   None
        self.method: MethodType | None      =   None
    
    def _save_embeddings(self, embeddings: EmbeddingData, save_dir = output_path("data")) -> None:
        dimension = len(next(iter(embeddings.values())))
        lib_hash = self._vocabulary_hash(dimension)
        file_name = f"embeddings-{lib_hash}-Dig-for-Fire.npz"
        file_path = output_path(save_dir, file_name)

        tokens, vectors = np.array(list(embeddings.keys())), np.stack(list(embeddings.values()))

        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(file_path, tokens=tokens, vectors=vectors)

    def _load_embeddings(self, embedding_dimension: int, save_dir=output_path("data")) -> EmbeddingData:
        lib_hash = self._vocabulary_hash(embedding_dimension)
        file_name = f"embeddings-{lib_hash}-Dig-for-Fire.npz"
        file_path = output_path(save_dir, file_name)
        if not os.path.exists(file_path):
            embeddings = self._compute_embeddings()
            self._save_embeddings(embeddings, save_dir=save_dir)
            return embeddings
        data = np.load(file_path)
        tokens, vectors = data["tokens"], data["vectors"]

        return {token: vectors[i] for i, token in enumerate(tokens)}
    
    @abstractmethod
    def _compute_embeddings(self, **kwargs) -> EmbeddingData:
        pass
    
    @abstractmethod
    def get_album_embeddings(self, album: AlbumData, **kwargs) -> np.ndarray:
        pass
    
    @abstractmethod
    def _build_vocabulary(self) -> list[str]:
        pass
    
    def _vocabulary_hash(self, embedding_dimension: int) -> str:
        library_data = [
            {
                "genres": list(get_album_genres(a).keys()),
                "tags": list(get_album_tags(a).keys())
            }
            
            for a in self.library
        ]

        payload = json.dumps({"library": library_data, "method": self.method, "token_type": self.token_type, "embedding_dimension": embedding_dimension}, sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()
    
    @abstractmethod
    def _smooth(self, embeddings: EmbeddingData, **kwargs) -> EmbeddingData:
        pass
            


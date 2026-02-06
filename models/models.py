from typing import Any, Literal
import numpy as np

AlbumData = dict[str, Any]
LibraryData = list[AlbumData]
MethodType = Literal["cooc", "svd", "pmi", "ppmi", "tag_clusters"]
TokenType = Literal["genre", "tag"]
EmbeddingData = dict[str, np.ndarray]

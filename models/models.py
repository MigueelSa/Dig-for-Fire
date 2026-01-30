from typing import Any, Literal

AlbumData = dict[str, Any]
LibraryData = list[AlbumData]
MethodType = Literal["cooc", "svd", "pmi", "ppmi"]

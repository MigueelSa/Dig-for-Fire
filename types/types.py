from typing import Any, Dict, List, TypeDict

class AlbumData(TypedDict):
    release_id: str
    title: str
    artist: list[str]
    date: str
    genres: list[str]
    tags: list[str]
    source: str

LibraryData = List[AlbumData]

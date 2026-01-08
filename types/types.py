from typing import Any, Dict, List, TypedDict

class AlbumData(TypedDict):
    release_id: str
    title: str
    artist: list[str]
    date: str
    genres: list[str]
    tags: list[str]
    source: str

LibraryData = List[AlbumData]

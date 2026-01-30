from models.models import AlbumData

def get_album_genres(album: AlbumData) -> dict[str, int]:
     return album.get("genres", {})

def get_album_tags(album: AlbumData) -> dict[str, int]:
    # all tags are present, so 0 is always assigned
    return {tag: 0 for tag in album.get("tags", [])}
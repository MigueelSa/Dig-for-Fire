import json

from digforfire.models.models import LibraryData

class SpotifyJSONConverter():
    @staticmethod
    def _convert(file: str) -> LibraryData:
        """Convert Spotify JSON data to a standard format."""
        with open(file) as f:
            library = json.load(f)
        albums = library.get("albums", [])
        converted_library = []
        for album in albums:
            converted_album = {
                "id": album.get("uri"),
                "title": album.get("album"),
                "artist": [album.get("artist")]
            }
            converted_library.append(converted_album)
        return converted_library
    
    @staticmethod
    def can_handle(file: str) -> bool:
        """Check if the file is a Spotify JSON file."""
        try:
            with open(file) as f:
                data = json.load(f)
            return "albums" in data
        except Exception:
            return False
    
    @staticmethod
    def save_converted_library(converted_library: LibraryData, output_file: str) -> None:
        """Save the converted library to a JSON file."""
        with open(output_file, "w") as f:
            json.dump(converted_library, f, indent=4)
        

    @staticmethod
    def convert_and_save(file: str) -> str:
        """Convert the file and save it to a JSON file."""
        output_file = file.replace(".json", "-converted.json")
        converted_library = SpotifyJSONConverter._convert(file)
        SpotifyJSONConverter.save_converted_library(converted_library, output_file)
        return output_file
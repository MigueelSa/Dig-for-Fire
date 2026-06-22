import json, pytest

from digforfire.libraries.converters import SpotifyJSONConverter


@pytest.fixture
def fake_files(tmp_path) -> tuple[str, str]:
    fake_file = tmp_path / "fake.json"
    fake_file.write_text("""{
                                "tracks": [
                                    {
                                        "artist": "The Mothers Of Invention",
                                        "album": "Freak Out!",
                                        "track": "You're Probably Wondering Why I'm Here",
                                        "uri": "spotify:track:6K7nz8CasqgyDGnkZOZuw8"
                                    },
                                    {
                                        "artist": "Penguin Cafe Orchestra",
                                        "album": "Music From The Penguin Cafe",
                                        "track": "The Sound Of Someone You Love Who's Going Away And It Doesn't Matter - 2008 Digital Remaster",
                                        "uri": "spotify:track:6K7nz8CasqgyDGnkZOZuw8"
                                    }
                                ],
                                "albums": [
                                    {
                                        "artist": "Boogarins",
                                        "album": "Lá Vem a Morte", 
                                        "uri": "spotify:album:1a2b3c4d5e6f7g8h9i0j"
                                    },
                                    {
                                        "artist": "Fausto",
                                        "album": "O despertar dos alquimistas",        
                                        "uri": "spotify:album:1a2b3c4d5e6f7g8h9i0j"
                                    },
                                    {
                                        "artist": "Bob Dylan",
                                        "album": "New Morning",
                                        "uri": "spotify:album:1a2b3c4d5e6f7g8h9i0j"
                                    }
                        
                                ],
                                "shows": [
                                    { 
                                        "name": "Democracy Now! Audio", 
                                        "publisher": "Democracy Now!", 
                                        "uri": "spotify:show:3cNrL5nALTDuWbRfabHeOG" 
                                    },
                                    {
                                        "name": "The Album Years",
                                        "publisher": "Steven Wilson & Tim Bowness",
                                        "uri": "spotify:show:36ZT08Ho2hVrttzi7s3FQA"
                                    }
                                ],
                                "episodes": [],
                                "bannedTracks": [],
                                "artists": [],
                                "bannedArtists": [],
                                "pageMatch": [],
                                "kallaxes": [],
                                "other": [],
                                "parentAllowedTracks": [],
                                "parentAllowedArtists": [],
                                "podcastChapters": []
                            }""")

    fake_converted_file = tmp_path / "fake-converted.json"
    fake_converted_file.write_text("""[
                                        {
                                            "id": "spotify:album:1a2b3c4d5e6f7g8h9i0j",
                                            "title": "Lá Vem a Morte",
                                            "artist": ["Boogarins"]
                                        },
                                        {
                                            "id": "spotify:album:1a2b3c4d5e6f7g8h9i0j",
                                            "title": "O despertar dos alquimistas",
                                            "artist": ["Fausto"]
                                        },
                                        {
                                            "id": "spotify:album:1a2b3c4d5e6f7g8h9i0j",
                                            "title": "New Morning",
                                            "artist": ["Bob Dylan"]
                                        }
                                    ]""")

    return fake_file, fake_converted_file


def test__can_handle(fake_files):
    fake_file, fake_converted_file = fake_files
    assert SpotifyJSONConverter.can_handle(fake_file) is True


def test__convert(fake_files):
    fake_file, fake_converted_file = fake_files
    assert SpotifyJSONConverter._convert(fake_file) == json.loads(
        fake_converted_file.read_text()
    )


def test__save_converted_library(fake_files):
    fake_file, fake_converted_file = fake_files
    assert SpotifyJSONConverter.convert_and_save(str(fake_file)) == str(
        fake_converted_file
    )

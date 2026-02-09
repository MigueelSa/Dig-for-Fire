from libraries.libraries import MusicBrainz

def add_album(title: str, artist: str, app_name: str, app_version: str, email: str) -> None:
    mb = MusicBrainz(app_name, app_version, email)
    mb.add_album(title, artist)

if __name__ == "__main__":
    import argparse

    app_name, app_version, email = "Dig-for-Fire", "0.1", "dig4fire-mail@proton.me"
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', type=str, required=True, help="Title of the album to add.")
    parser.add_argument('--artist', type=str, required=True, help="Artist of the album to add.")
    args = parser.parse_args()

    add_album(args.title, args.artist, app_name, app_version, email)
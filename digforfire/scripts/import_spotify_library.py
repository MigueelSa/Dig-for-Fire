from digforfire.libraries.libraries import Spotify

def import_spotify_library(client_id: str, client_secret:str, redirect_uri) -> None:
    spotify = Spotify(client_id, client_secret, redirect_uri)
    spotify.fetch_library()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=str, required=True, help="Spotify's client ID.")
    parser.add_argument('--client_secret', type=str, required=True, help="Spotify's client's secret.")
    parser.add_argument('--redirect_uri', type=str, required=True, help="Spotify's redirect URI. ")
    args = parser.parse_args()
    client_id, client_secret, redirect_uri = args.client_id, args.client_secret, args.redirect_uri

    import_spotify_library(client_id, client_secret, redirect_uri)
import argparse, logging, os, sys

from digforfire.libraries.libraries import MusicBrainz
from digforfire.recommender.recommend import Recommender
from digforfire.utils.paths import output_path
from digforfire.scripts.import_spotify_library import import_spotify_library
from digforfire.scripts.add_album import add_album

def main():
    parser = argparse.ArgumentParser(description="Dig-for-Fire: music recommendation engine")
    # spotify library import
    parser.add_argument('--client_id', type=str, required=False, help="Spotify's client ID.")
    parser.add_argument('--client_secret', type=str, required=False, help="Spotify's client's secret.")
    parser.add_argument('--redirect_uri', type=str, required=False, help="Spotify's redirect URI. ")
    # local library path
    parser.add_argument('--library_path', type=str, required=False, help="Path of the local library.")
    # add an album
    parser.add_argument('--add_album', nargs=2, metavar=('TITLE', 'ARTIST'), help="Add a single album")
    # recommend
    parser.add_argument('--recommend', action='store_true', help="Generate recommendations")
    parser.add_argument('--k', type=int, default=2, help="Number of recommendations")

    args = parser.parse_args()
    
    logging.basicConfig(filename='errors.log',
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.ERROR)

    app_name, app_version, email = "Dig-for-Fire", "0.1", "dig4fire-mail@proton.me"
    json_path = output_path("data", "MusicBrainz-Dig-for-Fire.json")

    if len(sys.argv) == 1:
        parser.print_help()
        return

    if args.client_id and args.client_secret and args.redirect_uri:
        client_id, client_secret, redirect_uri = args.client_id, args.client_secret, args.redirect_uri
        import_spotify_library(client_id, client_secret, redirect_uri)
        print("Library imported. If you want to enrich it with MusicBrainz run again with the option --library_path=data/Spotify-Dig-for-Fire.json")

    if args.library_path or args.add_album or args.recommend:
        if not os.path.exists(json_path):
            if args.library_path:
                MusicBrainz(app_name, app_version, email).fetch_library(os.path.abspath(args.library_path))
                
            else:
                logging.error("No library found. Please provide --library_path or Spotify credentials")
                print("Error: No library found. Use --library_path or Spotify credentials to import.")
                return
        else:
            if args.add_album:
                title, artist = args.add_album
                add_album(title, artist, app_name, app_version, email)

            if args.recommend:
                k = args.k
                rec = Recommender(json_path, email, k=k)
                recommendations = rec.recommend()
                sys.stdout.write('\r' + ' ' * 80 + '\r')
                sys.stdout.flush()
                print(recommendations)

if __name__ == "__main__":
    main()
import argparse, logging, os, sys, typer
from pathlib import Path

from digforfire.libraries.libraries import MusicBrainz
from digforfire.recommender.recommend import Recommender
from digforfire.utils.paths import output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library_path', type=str, required=False, default=None, help="Path of the local library.")
    args = parser.parse_args()
    
    logging.basicConfig(filename='errors.log',
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.ERROR)

    app_name, app_version, email = "Dig-for-Fire", "0.1", "dig4fire-mail@proton.me"

    json_path = output_path("data", "MusicBrainz-Dig-for-Fire.json")
    if not os.path.exists(json_path) and args.library_path is not None:
        library_path = os.path.abspath(args.library_path)
        musicbrainz = MusicBrainz(app_name, app_version, email)
        musicbrainz.fetch_library(library_path)
    elif not os.path.exists(json_path) and args.library_path is None:
        logging.error("You have not imported your library yet.")

    rec = Recommender(json_path, email)
    recommendations = rec.recommend()
    sys.stdout.write('\r' + ' ' * 80 + '\r')
    sys.stdout.flush()
    print(recommendations)

if __name__ == "__main__":
    main()
import argparse, logging, os, sys
from libraries.libraries import MusicBrainz
from recommender.recommend import Recommender
from utils.paths import output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library_path', type=str, required=True, help="Path of the local library.")
    args = parser.parse_args()
    
    logging.basicConfig(filename='errors.log',
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.ERROR)

    app_name, app_version, email = "Dig-for-Fire", "0.1", "dig4fire-mail@proton.me"
    library_path = os.path.abspath(args.library_path)

    json_path = output_path("data", "MusicBrainz-Dig-for-Fire.json")
    if not os.path.exists(json_path):
        musicbrainz = MusicBrainz(app_name, app_version, email)
        musicbrainz.fetch_library(library_path)

    rec = Recommender(json_path, email)
    recommendations = rec.recommend()
    print(recommendations)

if __name__ == "__main__":
    main()
import os
from dotenv import load_dotenv

from digforfire.config.config_base import get_env
from digforfire.config.config_base import ENV_PATH, project

load_dotenv(ENV_PATH)

APP_NAME = get_env("APP_NAME")
APP_VERSION = project["version"]
APP_EMAIL = get_env("APP_EMAIL")
APP_DESCRIPTION = project["description"]

API_HOST = get_env("API_HOST")
API_PORT = int(get_env("API_PORT"))

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
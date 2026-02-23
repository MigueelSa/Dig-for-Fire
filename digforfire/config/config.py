from digforfire.config import config_base

APP_NAME = config_base.get_env("APP_NAME")
APP_VERSION = config_base.project["version"]
APP_EMAIL = config_base.get_env("APP_EMAIL")
APP_DESCRIPTION = config_base.project["description"]

SPOTIFY_CLIENT_ID = config_base.get_env("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = config_base.get_env("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = config_base.get_env("SPOTIFY_REDIRECT_URI")

LASTFM_API_KEY = config_base.get_env("LASTFM_API_KEY")

API_HOST = config_base.get_env("API_HOST")
API_PORT = int(config_base.get_env("API_PORT"))
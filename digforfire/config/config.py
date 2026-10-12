from digforfire.config import config_base

APP_NAME = config_base.get_env("APP_NAME")
APP_VERSION = config_base.project["version"]
APP_EMAIL = config_base.get_env("APP_EMAIL")
APP_DESCRIPTION = config_base.project["description"]

LASTFM_API_KEY = config_base.get_env("LASTFM_API_KEY")

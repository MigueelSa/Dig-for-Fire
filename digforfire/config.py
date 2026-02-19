import os, tomli
from dotenv import load_dotenv
from pathlib import Path

pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
with pyproject_path.open("rb") as f:
    pyproject = tomli.load(f)
project = pyproject["project"]

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

def get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value

APP_NAME = get_env("APP_NAME")
APP_VERSION = project["version"]
APP_EMAIL = get_env("APP_EMAIL")
APP_DESCRIPTION = project["description"]

SPOTIFY_CLIENT_ID = get_env("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = get_env("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = get_env("SPOTIFY_REDIRECT_URI")

LASTFM_API_KEY = get_env("LASTFM_API_KEY")

API_HOST = get_env("API_HOST")
API_PORT = int(get_env("API_PORT"))

# --- contact info from pyproject.toml ---
contact = pyproject.get("tool", {}).get("digforfire", {}).get("contact", {})
import os, tomli
from dotenv import load_dotenv, set_key
from digforfire.utils.paths import resource_path

pyproject_path = resource_path("pyproject.toml")
with open(pyproject_path, "rb") as f:
    pyproject = tomli.load(f)
project = pyproject["project"]

ENV_PATH=resource_path(".env")
load_dotenv(ENV_PATH)

def get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value

def save_env(name: str, value: str):
    if not os.path.exists(ENV_PATH):
        with open(ENV_PATH, "w") as f: f.write("")
    set_key(ENV_PATH, name, value)
    os.environ[name] = value

# --- contact info from pyproject.toml ---
contact = pyproject.get("tool", {}).get("digforfire", {}).get("contact", {})
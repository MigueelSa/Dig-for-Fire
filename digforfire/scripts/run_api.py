import uvicorn, os, shutil

from digforfire.utils.paths import output_path, resource_path
from digforfire.config import config_base

def ensure_env():
    env_example_path = resource_path(".env.example")
    if not os.path.exists(config_base.ENV_PATH):
        shutil.copyfile(env_example_path, config_base.ENV_PATH)

def main() -> None:
    ensure_env()

    from digforfire.config import config_api

    source = output_path("data", "MusicBrainz-Dig-for-Fire.json")
    data_folder = output_path("digforfire", "static", "data")
    os.makedirs(data_folder, exist_ok=True)
    target = output_path(data_folder, "MusicBrainz-Dig-for-Fire.json")
    if os.path.exists(source) and not os.path.exists(target):
        os.symlink(source, target)
        
    uvicorn.run("digforfire.api:app", host=config_api.API_HOST, port=config_api.API_PORT, reload=True)

if __name__ == "__main__":
    main()
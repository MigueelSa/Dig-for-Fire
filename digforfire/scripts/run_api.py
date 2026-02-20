import uvicorn, os

from digforfire import config
from digforfire.utils.paths import output_path

def main() -> None:
    source = output_path("data", "MusicBrainz-Dig-for-Fire.json")
    data_folder = output_path("digforfire", "static", "data")
    os.makedirs(data_folder, exist_ok=True)
    target = output_path(data_folder, "MusicBrainz-Dig-for-Fire.json")
    if os.path.exists(source) and not os.path.exists(target):
        os.symlink(source, target)
        
    uvicorn.run("digforfire.api:app", host=config.API_HOST, port=config.API_PORT, reload=True)

if __name__ == "__main__":
    main()
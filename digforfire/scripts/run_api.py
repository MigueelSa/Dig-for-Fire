import uvicorn

from digforfire import config

def main() -> None:
    uvicorn.run("digforfire.api:app", host=config.API_HOST, port=config.API_PORT, reload=True)

if __name__ == "__main__":
    main()
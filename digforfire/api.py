from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates
import os, uuid
from dotenv import load_dotenv

from digforfire.recommender.recommend import Recommender
from digforfire.utils.paths import output_path, resource_path
from digforfire.utils.loading import ProgressTracker
from digforfire.libraries.libraries import MusicBrainz, Spotify
from digforfire.config import config_api, config_base   
from digforfire.recommender.history import HistoryManager

class AlbumResponse(BaseModel):
    id: str
    title: str
    artist: list[str]
    genres: list[str]
    tags: list[str]
    score: Optional[float] = 0.0
    cover_url: str

class LibraryResponse(BaseModel):
    library: List[AlbumResponse]

app = FastAPI(description=config_api.APP_DESCRIPTION)

json_path = output_path("data", "MusicBrainz-Dig-for-Fire.json")

tasks: dict[str, ProgressTracker] = {}


@app.get("/recommend")
def get_recommendations():
    if not os.path.exists(json_path):
        RedirectResponse(url="/user")
    try:
        load_dotenv(config_base.ENV_PATH, override=True)
        config_api.LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
        rec = Recommender(json_path, config_api.APP_EMAIL, config_api.LASTFM_API_KEY)
        results = rec.recommend()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    clean = [
        AlbumResponse(id=album["id"], title=album["title"], artist=album["artist"], 
                    genres=list(album["genres"].keys()), tags=album["tags"], score=album["score"],
                    cover_url = f"https://coverartarchive.org/release-group/{album['id']}/front-250")

        for album in results
        ]
    
    return LibraryResponse(library=clean)
    # Pydantic object → FastAPI automatically converts it to JSON string

@app.get("/library")
def get_library():
    try:
        rec = Recommender(json_path, config_api.APP_EMAIL, config_api.LASTFM_API_KEY)
        results = rec.library
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    clean = [
        AlbumResponse(id=album["id"], title=album["title"], artist=album["artist"], 
                    genres=list(album["genres"].keys()), tags=album["tags"],
                    cover_url = f"https://coverartarchive.org/release-group/{album['id']}/front-250")

        for album in results
        ]
    
    return LibraryResponse(library=clean)

@app.get("/history")
def get_history():
    results = HistoryManager.load_recommendations()
    clean = [
        AlbumResponse(id=album["id"], title=album["title"], artist=album["artist"], 
                    genres=list(album["genres"].keys()), tags=album["tags"], score=album["score"],
                    cover_url = f"https://coverartarchive.org/release-group/{album['id']}/front-250")

        for album in results
        ]
    
    return LibraryResponse(library=clean)

@app.post("/enrich-library")
async def enrich_library(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON files allowed.")
    
    content = await file.read()
    library_path = os.path.join("/tmp", file.filename)
    with open(library_path, "wb") as f:
        f.write(content)
    
    mb = MusicBrainz(app_name=config_api.APP_NAME, app_version=config_api.APP_VERSION, email=config_api.APP_EMAIL)
    mb.load_local_library(library_path)

    task_id = str(uuid.uuid4())
    tracker = ProgressTracker(total=len(mb.local_library))
    tasks[task_id] = tracker

    background_tasks.add_task(mb.fetch_library, library_path, progress=tracker)

    return {"status": "started", "task_id": task_id}

@app.get("/progress/{task_id}")
def get_progress(task_id: str):
    tracker = tasks.get(task_id)
    if not tracker:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "current": tracker.current,
        "total": tracker.total,
        "ratio": tracker.ratio,
        "eta": tracker.eta
    }

@app.post("/user/spotify/save-credentials")
async def save_spotify_credentials(request: Request):
    data = await request.json()
    client_id, client_secret, lastfm_api_key = data.get("client_id"), data.get("client_secret"), data.get("lastfm_api_key")

    if not os.path.exists(config_base.ENV_PATH):
        with open(config_base.ENV_PATH, "w") as f: f.write("")
    
    if client_id and client_secret and lastfm_api_key:
        config_base.save_env("SPOTIFY_CLIENT_ID", client_id)
        config_base.save_env("SPOTIFY_CLIENT_SECRET", client_secret)
        config_base.save_env("LASTFM_API_KEY", lastfm_api_key)

@app.get("/user/spotify/credentials-check")
def spotify_credentials_check():
    load_dotenv(config_base.ENV_PATH, override=True)
    config_api.SPOTIFY_CLIENT_ID, config_api.SPOTIFY_CLIENT_SECRET, config_api.LASTFM_API_KEY = os.getenv("SPOTIFY_CLIENT_ID"), os.getenv("SPOTIFY_CLIENT_SECRET"), os.getenv("LASTFM_API_KEY")
    ok = bool(config_api.SPOTIFY_CLIENT_ID and config_api.SPOTIFY_CLIENT_SECRET and config_api.LASTFM_API_KEY)
    return {"ok": ok}
    
@app.get("/user/spotify/import-library")
def import_spotify_library():
    spotify = Spotify(config_api.SPOTIFY_CLIENT_ID, config_api.SPOTIFY_CLIENT_SECRET, config_api.SPOTIFY_REDIRECT_URI)
    spotify.fetch_library()


templates = Jinja2Templates(directory=resource_path("digforfire", "templates"))
app.mount("/static", StaticFiles(directory=resource_path("digforfire", "static"), follow_symlink=True), name="static")

@app.get("/", response_class=HTMLResponse)
def homepage(request: Request):
    if os.path.exists(json_path):
        return templates.TemplateResponse("recommend.html", {"request": request, "contact": config_base.contact})
    else:
        return RedirectResponse(url="/user")

@app.get("/user", response_class=HTMLResponse)
def userpage(request: Request):
    return templates.TemplateResponse("user.html", {"request": request, "contact": config_base.contact})

@app.get("/user/spotify/setup", response_class=HTMLResponse)
def userpage(request: Request):
    return templates.TemplateResponse("spotify_setup.html", {"request": request, "contact": config_base.contact})

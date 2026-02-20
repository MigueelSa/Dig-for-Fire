from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates
import os



from digforfire.recommender.recommend import Recommender
from digforfire.utils.paths import output_path, resource_path
from digforfire.libraries.libraries import MusicBrainz
from digforfire import config

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

app = FastAPI(description=config.APP_DESCRIPTION)

app_name = config.APP_NAME
app_version = config.APP_VERSION
json_path = output_path("data", "MusicBrainz-Dig-for-Fire.json")
email = config.APP_EMAIL

@app.get("/recommend")
def get_recommendations():
    try:
        rec = Recommender(json_path, email)
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
        rec = Recommender(json_path, email)
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
def get_library():
    try:
        rec = Recommender(json_path, email)
        results = rec.recommendation_history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
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
    
    mb = MusicBrainz(app_name=app_name, app_version=app_version, email=email)
    background_tasks.add_task(mb.fetch_library, library_path)

    return {"status": "Library imported and enriched successfully!"}


templates = Jinja2Templates(directory=resource_path("digforfire", "templates"))
app.mount("/static", StaticFiles(directory=resource_path("digforfire", "static"), follow_symlink=True), name="static")

@app.get("/", response_class=HTMLResponse)
def homepage(request: Request):
    if os.path.exists(json_path):
        return templates.TemplateResponse("recommend.html", {"request": request, "contact": config.contact})
    else:
        return RedirectResponse(url="/user")

@app.get("/user", response_class=HTMLResponse)
def userpage(request: Request):
    return templates.TemplateResponse("user.html", {"request": request, "contact": config.contact})

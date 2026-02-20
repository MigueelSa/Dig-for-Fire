from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates



from digforfire.recommender.recommend import Recommender
from digforfire.utils.paths import output_path, resource_path
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

json_path = output_path("data", "MusicBrainz-Dig-for-Fire.json")
email = config.APP_EMAIL
rec = Recommender(json_path, email)

@app.get("/recommend")
def get_recommendations():
    try:
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

templates = Jinja2Templates(directory=resource_path("digforfire", "templates"))
app.mount("/static", StaticFiles(directory=resource_path("digforfire", "static"), follow_symlink=True), name="static")

@app.get("/", response_class=HTMLResponse)
def homepage(request: Request):
    return templates.TemplateResponse("recommend.html", {"request": request, "contact": config.contact})

@app.get("/user", response_class=HTMLResponse)
def userpage(request: Request):
    return templates.TemplateResponse("user.html", {"request": request, "contact": config.contact})

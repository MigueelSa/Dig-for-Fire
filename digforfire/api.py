from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional


from digforfire.recommender.recommend import Recommender
from digforfire.utils.paths import output_path
from digforfire import config

class AlbumResponse(BaseModel):
    id: str
    title: str
    artist: list[str]
    genres: list[str]
    tags: list[str]
    score: float

class RecommendationResponse(BaseModel):
    recommendations: List[AlbumResponse]

app = FastAPI()
json_path = output_path("data", "MusicBrainz-Dig-for-Fire.json")
email = config.APP_EMAIL

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/recommend")
def get_recommendations():
    results = Recommender(json_path, email).recommend()
    clean = [
        AlbumResponse(id=album["id"], title=album["title"], artist=album["artist"], 
                    genres=list(album["genres"].keys()), tags=album["tags"], score=album["score"])

        for album in results
        ]
    
    return RecommendationResponse(recommendations=clean)

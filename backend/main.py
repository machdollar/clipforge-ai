import os
import re
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

app = FastAPI(
    title="ClipForge AI API",
    description="Backend for AI-powered video highlight generation",
    version="0.1.0",
)


class AnalyzeRequest(BaseModel):
    youtube_url: HttpUrl


def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host == "youtu.be":
        video_id = parsed.path.strip("/")
    elif host == "youtube.com" or host.endswith(".youtube.com"):
        if parsed.path == "/watch":
            from urllib.parse import parse_qs
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/shorts/", "/live/", "/embed/")):
            video_id = parsed.path.split("/")[2]
        else:
            video_id = ""
    else:
        video_id = ""

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid YouTube video URL.",
        )

    return video_id


@app.get("/")
def home():
    return {
        "name": "ClipForge AI API",
        "status": "online",
        "message": "Backend is running. AI analysis is not connected yet.",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    video_id = extract_video_id(str(request.youtube_url))

    # This first version validates the URL only.
    # Transcript retrieval, AI highlight selection, and clip rendering
    # will be added in later steps.
    return {
        "status": "accepted",
        "video_id": video_id,
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "highlights": [],
        "message": (
            "The URL is valid. AI analysis and clip generation "
            "are not connected yet."
        ),
    }

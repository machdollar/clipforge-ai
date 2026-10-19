import os
import re
from urllib.parse import urlparse, parse_qs

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from openai import OpenAI, APIError

app = FastAPI(
    title="ClipForge AI API",
    description="Backend for AI-powered video highlight generation",
    version="0.2.0",
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
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/shorts/", "/live/", "/embed/")):
            parts = parsed.path.split("/")
            video_id = parts[2] if len(parts) > 2 else ""
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
        "message": "Backend is running.",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ai-test")
def ai_test():
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key is not configured on the server.",
        )

    try:
        client = OpenAI()
        response = client.responses.create(
            model="gpt-5.5",
            input="Reply with exactly: ClipForge AI connection successful.",
            max_output_tokens=30,
        )
        return {
            "status": "success",
            "message": response.output_text,
        }
    except APIError:
        # Do not expose credentials or detailed provider errors.
        raise HTTPException(
            status_code=502,
            detail="The OpenAI request failed. Check API access, model availability, and billing in your provider account.",
        )


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    video_id = extract_video_id(str(request.youtube_url))
    return {
        "status": "accepted",
        "video_id": video_id,
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "highlights": [],
        "message": (
            "The URL is valid. AI highlight selection and clip rendering "
            "are not implemented yet."
        ),
        }

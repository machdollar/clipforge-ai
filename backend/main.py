import os
import re
from urllib.parse import urlparse, parse_qs

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from openai import OpenAI, APIError
from youtube_transcript_api import YouTubeTranscriptApi

app = FastAPI(
    title="ClipForge AI API",
    description="AI-powered video highlight analysis",
    version="0.3.0",
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
        raise HTTPException(status_code=400, detail="Please provide a valid YouTube URL.")

    return video_id


@app.get("/")
def home():
    return {"name": "ClipForge AI API", "status": "online"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ai-test")
def ai_test():
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OpenAI API key is not configured.")

    try:
        response = OpenAI().responses.create(
            model="gpt-5.5",
            input="Reply with exactly: ClipForge AI connection successful.",
            max_output_tokens=30,
        )
        return {"status": "success", "message": response.output_text}
    except APIError:
        raise HTTPException(status_code=502, detail="OpenAI request failed. Check model access and API billing.")


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    video_id = extract_video_id(str(request.youtube_url))

    try:
        transcript_items = YouTubeTranscriptApi().fetch(video_id)
        transcript = [
            {
                "start": float(item.start),
                "duration": float(item.duration),
                "text": item.text,
            }
            for item in transcript_items
        ]
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="A transcript could not be retrieved for this video. It may be unavailable or restricted. Try a video with captions enabled.",
        )

    if not transcript:
        raise HTTPException(status_code=422, detail="The transcript is empty.")

    # Limit transcript size and retain timestamps.
    transcript = transcript[:1200]
    transcript_text = "\n".join(
        f"[{item['start']:.1f}s] {item['text']}"
        for item in transcript
    )

    try:
        response = OpenAI().responses.create(
            model="gpt-5.5",
            instructions=(
                "You are a video editor. Identify up to 5 engaging short-form "
                "video moments from the supplied transcript. Only use timestamps "
                "present in the transcript. Return valid JSON with a highlights "
                "array. Each highlight must have start_seconds, end_seconds, "
                "title, and reason. Choose segments around 15 to 60 seconds when "
                "the transcript supports it. Do not invent transcript content."
            ),
            input=transcript_text,
            max_output_tokens=1200,
        )

        import json
        result_text = response.output_text
        result = json.loads(result_text)
        highlights = result.get("highlights", [])

        return {
            "status": "success",
            "video_id": video_id,
            "video_url": f"https://www.youtube.com/watch?v={video_id}",
            "highlights": highlights[:5],
            "message": "Transcript analyzed. These are suggested moments, not rendered clips.",
        }
    except (APIError, ValueError):
        raise HTTPException(
            status_code=502,
            detail="AI analysis failed. Check model access and try again.",
        )

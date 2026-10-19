import json
import os
import re
from urllib.parse import urlparse, parse_qs

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, HttpUrl
from openai import OpenAI, APIError
from youtube_transcript_api import YouTubeTranscriptApi

app = FastAPI(
    title="ClipForge AI API",
    description="AI-powered video and audio highlight analysis",
    version="0.4.0",
)

MAX_AUDIO_BYTES = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {
    ".mp3", ".m4a", ".wav", ".webm", ".mp4",
    ".mpeg", ".mpga", ".oga", ".ogg", ".flac"
}


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


def detect_highlights(transcript_text: str):
    try:
        response = OpenAI().responses.create(
            model="gpt-5.5",
            instructions=(
                "Identify up to 15 engaging short-form moments from this "
                "timestamped transcript. Only use supplied timestamps; do not "
                "invent content. Return valid JSON with a highlights array. "
                "Each item needs numeric start_seconds, end_seconds, title, "
                "and reason. Prefer 15-60 second moments where supported."
            ),
            input=transcript_text,
            max_output_tokens=3000,
        )
        result = json.loads(response.output_text)
        highlights = result.get("highlights", [])
        if not isinstance(highlights, list):
            raise ValueError("Invalid highlights format")
        return highlights[:15]
    except (APIError, ValueError):
        raise HTTPException(
            status_code=502,
            detail="AI analysis failed. Check model access and API billing."
        )


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
        raise HTTPException(
            status_code=502,
            detail="OpenAI request failed. Check model access and API billing."
        )


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    video_id = extract_video_id(str(request.youtube_url))

    try:
        items = YouTubeTranscriptApi().fetch(video_id)
        transcript = [
            {"start": float(item.start), "text": item.text}
            for item in items
        ]
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Could not retrieve this video's transcript. Try /upload-audio with an authorized audio file."
        )

    if not transcript:
        raise HTTPException(status_code=422, detail="The transcript is empty.")

    transcript_text = "\n".join(
        f"[{item['start']:.1f}s] {item['text']}"
        for item in transcript[:1200]
    )
    highlights = detect_highlights(transcript_text)

    return {
        "status": "success",
        "video_id": video_id,
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "highlights": highlights,
        "message": "Suggested moments only; video clips have not been rendered."
    }


@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OpenAI API key is not configured.")

    filename = os.path.basename(file.filename or "audio")
    extension = os.path.splitext(filename)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported format. Use MP3, M4A, WAV, WebM, MP4, OGG, OGA, FLAC, or MPEG."
        )

    audio_bytes = await file.read(MAX_AUDIO_BYTES + 1)
    await file.close()

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Maximum upload size is 20 MB.")

    try:
        transcription = OpenAI().audio.transcriptions.create(
            model="whisper-1",
            file=(
                filename,
                audio_bytes,
                file.content_type or "application/octet-stream"
            ),
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    except APIError as exc:
    error_status = getattr(exc, "status_code", None)
    error_code = getattr(exc, "code", None)
    error_type = type(exc).__name__

    print(
        f"Audio transcription error: type={error_type}, "
        f"status={error_status}, code={error_code}"
    )

    raise HTTPException(
        status_code=502,
        detail={
            "message": "OpenAI transcription request failed.",
            "error_type": error_type,
            "openai_status": error_status,
            "openai_code": error_code,
            "hint": "Check Render logs and OpenAI API billing/model access."
        }
    )

    segments = getattr(transcription, "segments", None) or []
    if not segments:
        raise HTTPException(
            status_code=422,
            detail="No timestamped speech segments were returned. Try another audio file."
        )

    transcript_text = "\n".join(
        f"[{float(segment.start):.1f}s] {segment.text.strip()}"
        for segment in segments
        if getattr(segment, "text", "").strip()
    )

    if not transcript_text:
        raise HTTPException(status_code=422, detail="No speech could be transcribed.")

    highlights = detect_highlights(transcript_text)

    return {
        "status": "success",
        "filename": filename,
        "transcript": getattr(transcription, "text", ""),
        "highlights": highlights,
        "message": "Audio transcribed and analyzed. Suggested moments only; no clips rendered."
            }
    

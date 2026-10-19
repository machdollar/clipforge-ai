
import json
import os
import re
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import APIError, OpenAI
from pydantic import BaseModel, HttpUrl
from youtube_transcript_api import YouTubeTranscriptApi


app = FastAPI(
    title="ClipForge AI API",
    description="AI-powered video and audio highlight analysis",
    version="0.5.0",
)

MAX_AUDIO_BYTES = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".mp3", ".m4a", ".wav", ".webm", ".mp4",
    ".mpeg", ".mpga", ".oga", ".ogg", ".flac",
}


class AnalyzeRequest(BaseModel):
    youtube_url: HttpUrl


def get_openai_client():
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key is not configured on the server.",
        )
    return OpenAI()


def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    video_id = ""

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host == "youtube.com" or host.endswith(".youtube.com"):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif parsed.path.startswith(("/shorts/", "/live/", "/embed/")):
            parts = parsed.path.split("/")
            if len(parts) > 2:
                video_id = parts[2]

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid YouTube video URL.",
        )

    return video_id


def detect_highlights(transcript_text: str) -> list:
    client = get_openai_client()

    try:
        response = client.responses.create(
            model="gpt-5.5",
            instructions=(
                "Analyze the supplied timestamped transcript and identify "
                "up to 15 engaging moments suitable for short-form videos. "
                "Return a JSON object with a 'highlights' array. "
                "Each highlight must contain numeric start_seconds, "
                "numeric end_seconds, a short title, and a reason. "
                "Use only timestamps supported by the transcript. "
                "Do not invent events or timestamps. Prefer moments "
                "15 to 60 seconds long when supported by the source. "
                "Order the moments by start_seconds."
            ),
            input=transcript_text,
            max_output_tokens=4000,
        )

        result = json.loads(response.output_text)
        highlights = result.get("highlights", [])

        if not isinstance(highlights, list):
            raise ValueError("The highlights field is not a list.")

        clean_highlights = []

        for item in highlights[:15]:
            if not isinstance(item, dict):
                continue

            start = item.get("start_seconds")
            end = item.get("end_seconds")
            title = item.get("title")
            reason = item.get("reason")

            if (
                isinstance(start, (int, float))
                and not isinstance(start, bool)
                and isinstance(end, (int, float))
                and not isinstance(end, bool)
                and 0 <= start < end
                and isinstance(title, str)
                and isinstance(reason, str)
            ):
                clean_highlights.append({
                    "start_seconds": float(start),
                    "end_seconds": float(end),
                    "title": title[:200],
                    "reason": reason[:500],
                })

        return clean_highlights

    except (APIError, ValueError, TypeError) as exc:
        print(
            "Highlight analysis failed: "
            f"{type(exc).__name__}: {str(exc)[:500]}"
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "AI highlight analysis failed. Check the Render logs, "
                "OpenAI model access, and API billing."
            ),
        ) from exc


@app.get("/")
def home():
    return {
        "name": "ClipForge AI API",
        "status": "online",
        "version": "0.5.0",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ai-test")
def ai_test():
    client = get_openai_client()

    try:
        response = client.responses.create(
            model="gpt-5.5",
            input="Reply with exactly: ClipForge AI connection successful.",
            max_output_tokens=30,
        )
        return {
            "status": "success",
            "message": response.output_text,
        }

    except APIError as exc:
        print(
            f"AI test failed: {type(exc).__name__}: "
            f"status={getattr(exc, 'status_code', None)}, "
            f"code={getattr(exc, 'code', None)}"
        )
        raise HTTPException(
            status_code=502,
            detail="OpenAI request failed. Check Render logs and API access.",
        ) from exc


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    video_id = extract_video_id(str(request.youtube_url))

    try:
        items = YouTubeTranscriptApi().fetch(video_id)
        transcript = [
            {
                "start": float(item.start),
                "text": str(item.text),
            }
            for item in items
        ]

    except Exception as exc:
        print(
            f"YouTube transcript retrieval failed: "
            f"{type(exc).__name__}: {str(exc)[:300]}"
        )
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not retrieve this video's transcript. "
                "Try a video with accessible captions or use "
                "the /upload-audio endpoint with an audio file "
                "you are authorized to use."
            ),
        ) from exc

    if not transcript:
        raise HTTPException(
            status_code=422,
            detail="The transcript is empty.",
        )

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
        "count": len(highlights),
        "message": (
            "Suggested moments only. Video clips have not been rendered."
        ),
    }


@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or "audio")
    extension = os.path.splitext(filename)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        await file.close()
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported format. Use MP3, M4A, WAV, WebM, MP4, "
                "OGG, OGA, FLAC, or MPEG."
            ),
        )

    try:
        audio_bytes = await file.read(MAX_AUDIO_BYTES + 1)
    finally:
        await file.close()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Maximum upload size is 20 MB.",
        )

    client = get_openai_client()

    try:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=(
                filename,
                audio_bytes,
                file.content_type or "application/octet-stream",
            ),
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    except APIError as exc:
        error_status = getattr(exc, "status_code", None)
        error_code = getattr(exc, "code", None)

        print(
            "Audio transcription failed: "
            f"type={type(exc).__name__}, "
            f"status={error_status}, code={error_code}, "
            f"message={str(exc)[:500]}"
        )

        raise HTTPException(
            status_code=502,
            detail={
                "message": "OpenAI audio transcription failed.",
                "error_type": type(exc).__name__,
                "openai_status": error_status,
                "openai_code": error_code,
                "hint": (
                    "Check Render logs, OpenAI API billing, "
                    "model access, and the uploaded file."
                ),
            },
        ) from exc

    segments = getattr(transcription, "segments", None) or []

    transcript_parts = []

    for segment in segments:
        if isinstance(segment, dict):
            start = segment.get("start")
            text = segment.get("text", "")
        else:
            start = getattr(segment, "start", None)
            text = getattr(segment, "text", "")

        if start is not None and str(text).strip():
            transcript_parts.append(
                f"[{float(start):.1f}s] {str(text).strip()}"
            )

    if not transcript_parts:
        raise HTTPException(
            status_code=422,
            detail=(
                "No timestamped speech segments were returned. "
                "Try another audio file containing speech."
            ),
        )

    transcript_text = "\n".join(transcript_parts)
    highlights = detect_highlights(transcript_text)

    return {
        "status": "success",
        "filename": filename,
        "transcript": getattr(transcription, "text", ""),
        "highlights": highlights,
        "count": len(highlights),
        "message": (
            "Audio transcribed and analyzed. "
            "Suggested moments only; video clips have not been rendered."
        ),
                        }
                

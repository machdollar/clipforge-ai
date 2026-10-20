import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi


APP_VERSION = "1.2.0"

app = FastAPI(
    title="ClipForge AI API",
    description=(
        "Transcript highlight suggestions and YouTube transcript importing. "
        "Uses local scoring rules and does not require paid AI API calls."
    ),
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://machdollar.github.io"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class TranscriptRequest(BaseModel):
    transcript: str = Field(..., min_length=20, max_length=100000)


class YouTubeRequest(BaseModel):
    url: str = Field(..., min_length=8, max_length=2048)


TIMESTAMP_TOKEN = (
    r"\d{1,3}:\d{2}:\d{2}(?:\.\d+)?"
    r"|"
    r"\d{1,4}:\d{2}(?:\.\d+)?"
    r"|"
    r"\d+(?:\.\d+)?\s*(?:seconds?|secs?|s)"
    r"|"
    r"\d+\s*m(?:in(?:ute)?s?)?"
    r"(?:\s*\d+(?:\.\d+)?\s*s?)?"
)

TIMESTAMP_PATTERN = re.compile(
    r"(?<!\S)(?:[\[(]\s*)?"
    r"(?P<timestamp>" + TIMESTAMP_TOKEN + r")"
    r"\s*[\])}]?\s*(?:[-–—|]\s*)?",
    re.IGNORECASE,
)

WORD_PATTERN = re.compile(r"[a-zA-Z0-9']+")

KEYWORD_GROUPS = {
    "hook": {
        "secret", "surprising", "shocking", "unexpected",
        "nobody", "everyone", "mistake", "truth", "warning",
        "finally", "revealed", "discover", "imagine",
    },
    "learning": {
        "learn", "lesson", "tip", "guide", "explain",
        "solution", "method", "strategy", "step", "steps",
        "important", "remember", "understand", "because",
    },
    "emotion": {
        "amazing", "incredible", "exciting", "powerful",
        "changed", "success", "fail", "failure", "problem",
        "challenge", "best", "worst", "favorite",
    },
    "action": {
        "start", "stop", "try", "use", "make", "build",
        "create", "avoid", "improve", "change", "help",
    },
}

FILLER_WORDS = set(
    "a an and are as at be but by for from had has have he "
    "her here him his i if in into is it its just me my of "
    "on or our she so that the their them there they this to "
    "us was we were what when where which who will with you your"
    .split()
)


def parse_timestamp(value: str) -> float:
    value = value.strip().lower().strip("[](){} ")

    if re.fullmatch(r"\d{1,3}:\d{2}:\d{2}(?:\.\d+)?", value):
        hours, minutes, seconds = value.split(":")
        if int(minutes) >= 60 or float(seconds) >= 60:
            raise ValueError("Invalid timestamp")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    if re.fullmatch(r"\d{1,4}:\d{2}(?:\.\d+)?", value):
        minutes, seconds = value.split(":")
        if float(seconds) >= 60:
            raise ValueError("Invalid timestamp")
        return int(minutes) * 60 + float(seconds)

    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)",
        value,
    )
    if match:
        return float(match.group(1))

    match = re.fullmatch(
        r"(\d+)\s*m(?:in(?:ute)?s?)?"
        r"(?:\s*(\d+(?:\.\d+)?)\s*s?)?",
        value,
    )
    if match:
        seconds = float(match.group(2) or 0)
        if seconds >= 60:
            raise ValueError("Invalid timestamp")
        return int(match.group(1)) * 60 + seconds

    raise ValueError("Unsupported timestamp format")


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    remaining = total % 60

    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining:02d}"

    return f"{minutes:02d}:{remaining:02d}"


def parse_transcript(
    transcript: str,
) -> tuple[list[dict[str, Any]], int]:
    entries = []
    skipped_lines = 0

    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue

        matches = list(TIMESTAMP_PATTERN.finditer(line))

        if not matches:
            skipped_lines += 1
            continue

        found = 0

        for index, match in enumerate(matches):
            try:
                timestamp = parse_timestamp(match.group("timestamp"))
            except (ValueError, OverflowError):
                skipped_lines += 1
                continue

            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(line)
            )

            text = line[match.end():end].strip()
            text = text.strip(" -–—|:;,.")
            text = re.sub(r"\s+", " ", text)

            if not text:
                skipped_lines += 1
                continue

            entries.append({
                "timestamp": round(timestamp, 2),
                "text": text,
            })
            found += 1

        if not found:
            skipped_lines += 1

    entries.sort(key=lambda item: item["timestamp"])

    merged = []
    for entry in entries:
        if merged and merged[-1]["timestamp"] == entry["timestamp"]:
            merged[-1]["text"] += " " + entry["text"]
        else:
            merged.append(entry.copy())

    return merged, skipped_lines


def score_entry(text: str) -> float:
    words = [word.lower() for word in WORD_PATTERN.findall(text)]
    if not words:
        return 0.0

    unique = set(words)
    score = 0.0

    for group in KEYWORD_GROUPS.values():
        score += len(unique.intersection(group)) * 1.5

    meaningful = [
        word for word in words
        if word not in FILLER_WORDS and len(word) > 2
    ]
    score += min(len(meaningful), 20) * 0.12

    if len(words) < 4:
        score -= 1.0
    if "?" in text:
        score += 0.8
    if "!" in text:
        score += 0.5
    if text.rstrip().endswith((".", "!", "?", "…")):
        score += 0.4

    return round(max(score, 0.0), 3)


def make_title(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" .,!?:;—-")
    if not text:
        return "Potential highlight"

    words = text.split()
    title = " ".join(words[:8])

    if len(words) > 8:
        title += "…"
    if len(title) > 70:
        title = title[:67].rstrip() + "..."

    return title[0].upper() + title[1:]


def explain_score(text: str) -> str:
    words = {word.lower() for word in WORD_PATTERN.findall(text)}
    reasons = []

    if words.intersection(KEYWORD_GROUPS["hook"]):
        reasons.append("contains a possible attention-grabbing hook")
    if words.intersection(KEYWORD_GROUPS["learning"]):
        reasons.append("contains a useful lesson or explanation")
    if words.intersection(KEYWORD_GROUPS["emotion"]):
        reasons.append("contains expressive or impactful language")
    if words.intersection(KEYWORD_GROUPS["action"]):
        reasons.append("contains actionable advice")
    if "?" in text:
        reasons.append("uses a question that may engage viewers")

    if not reasons:
        reasons.append("selected from the available transcript segments")

    return "; ".join(reasons).capitalize() + "."


def choose_highlights(
    entries: list[dict[str, Any]],
    limit: int = 15,
) -> list[dict[str, Any]]:
    ranked = []

    for entry in entries:
        ranked.append({
            "timestamp": entry["timestamp"],
            "text": entry["text"],
            "score": score_entry(entry["text"]),
        })

    ranked.sort(
        key=lambda item: (item["score"], len(item["text"])),
        reverse=True,
    )

    selected = []

    # First prefer stronger suggestions that are not too close together.
    for candidate in ranked:
        if len(selected) >= limit:
            break
        if candidate["score"] <= 0:
            continue
        if any(
            abs(candidate["timestamp"] - item["timestamp"]) < 12
            for item in selected
        ):
            continue
        selected.append(candidate)

    # Fill remaining slots for short transcripts.
    selected_times = {item["timestamp"] for item in selected}
    for candidate in ranked:
        if len(selected) >= limit:
            break
        if candidate["timestamp"] not in selected_times:
            selected.append(candidate)
            selected_times.add(candidate["timestamp"])

    selected.sort(key=lambda item: item["timestamp"])
    results = []

    for index, item in enumerate(selected):
        start = item["timestamp"]

        if index + 1 < len(selected):
            end = max(start + 5, selected[index + 1]["timestamp"])
        else:
            end = start + 30

        results.append({
            "title": make_title(item["text"]),
            "start_seconds": round(start, 2),
            "end_seconds": round(end, 2),
            "timestamp": format_timestamp(start),
            "score": item["score"],
            "reason": explain_score(item["text"]),
            "excerpt": item["text"],
        })

    return results


def extract_youtube_video_id(url: str) -> str:
    value = url.strip()
    parsed = urlparse(
        value if "://" in value else "https://" + value
    )
    host = (parsed.hostname or "").lower()
    video_id = ""

    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]

    elif host in {
        "youtube.com", "www.youtube.com", "m.youtube.com",
        "music.youtube.com", "youtube-nocookie.com",
        "www.youtube-nocookie.com",
    }:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in {
                "shorts", "embed", "live", "v"
            }:
                video_id = parts[1]

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise HTTPException(
            status_code=400,
            detail="Please enter a valid YouTube video URL.",
        )

    return video_id


def get_highlights_response(
    transcript: str,
    mode: str,
    video_id: str | None = None,
) -> dict[str, Any]:
    entries, skipped = parse_transcript(transcript)
    highlights = choose_highlights(entries)

    if not highlights:
        raise HTTPException(
            status_code=400,
            detail=(
                "No usable timestamped transcript segments were found. "
                "Check that the transcript includes timestamps."
            ),
        )

    response = {
        "status": "success",
        "mode": mode,
        "count": len(highlights),
        "highlights": highlights,
        "skipped_lines": skipped,
        "message": (
            "Highlight suggestions generated. They are heuristic suggestions, "
            "not verified viral moments or actual video clips."
        ),
    }

    if video_id:
        response["video_id"] = video_id

    return response


@app.get("/")
def home() -> dict[str, Any]:
    return {
        "name": "ClipForge AI API",
        "version": APP_VERSION,
        "status": "online",
        "message": "ClipForge AI backend is running.",
        "endpoints": {
            "health": "/health",
            "transcript_highlights": "/transcript-highlights",
            "youtube_highlights": "/youtube-transcript-highlights",
            "documentation": "/docs",
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "version": APP_VERSION,
        "mode": "local transcript scoring",
    }


@app.post("/transcript-highlights")
def transcript_highlights(
    request: TranscriptRequest,
) -> dict[str, Any]:
    return get_highlights_response(
        request.transcript.strip(),
        "local transcript scoring",
    )


@app.post("/youtube-transcript-highlights")
def youtube_transcript_highlights(
    request: YouTubeRequest,
) -> dict[str, Any]:
    video_id = extract_youtube_video_id(request.url)

    try:
        fetched = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["en"],
        )

        lines = []
        for snippet in fetched:
            text = re.sub(r"\s+", " ", str(snippet.text)).strip()
            if text:
                lines.append(
                    f"{format_timestamp(float(snippet.start))} {text}"
                )

    except Exception as exc:
        error_name = type(exc).__name__

        if error_name in {"TranscriptsDisabled", "NoTranscriptFound"}:
            message = (
                "No accessible English transcript was found. Try a video "
                "with captions or paste a transcript manually."
            )
        elif error_name in {"VideoUnavailable", "InvalidVideoId"}:
            message = "This YouTube video is unavailable or its URL is invalid."
        else:
            message = (
                "Could not retrieve this transcript. Captions may be "
                "unavailable or YouTube may be blocking automated access. "
                "You can still paste a transcript manually."
            )

        raise HTTPException(status_code=400, detail=message) from None

    transcript = "\n".join(lines)

    if len(transcript) < 20:
        raise HTTPException(
            status_code=400,
            detail="The retrieved transcript is empty or too short.",
        )

    return get_highlights_response(
        transcript,
        "YouTube transcript + local transcript scoring",
        video_id,
        )
                

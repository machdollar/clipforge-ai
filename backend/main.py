import re
from collections import Counter
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi


APP_VERSION = "2.0.0"
MAX_HIGHLIGHTS = 15
CLIP_LENGTHS = (15, 25, 35, 45)
MIN_CLIP_SECONDS = 10
MAX_TRANSCRIPT_CHARS = 100_000

app = FastAPI(
    title="ClipForge AI API",
    description=(
        "Free, rule-based transcript clipping, creator tools, "
        "editing plans, and transcript insights."
    ),
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://machdollar.github.io",
        "http://localhost",
        "http://127.0.0.1",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class TranscriptRequest(BaseModel):
    transcript: str = Field(
        ..., min_length=20, max_length=MAX_TRANSCRIPT_CHARS
    )


class YouTubeRequest(BaseModel):
    url: str = Field(..., min_length=8, max_length=2048)


class EditPlanRequest(BaseModel):
    start_seconds: float = Field(..., ge=0)
    end_seconds: float = Field(..., gt=0)
    aspect_ratio: str = Field(default="9:16")
    captions: bool = True
    crop_to_vertical: bool = True


class CreatorRequest(BaseModel):
    transcript: str = Field(
        ..., min_length=10, max_length=MAX_TRANSCRIPT_CHARS
    )
    platform: str = Field(default="YouTube Shorts", max_length=50)


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

KEYWORDS = {
    "hook": {
        "secret", "surprising", "shocking", "unexpected", "mistake",
        "truth", "warning", "revealed", "discover", "actually", "never",
        "always", "crazy", "wild", "unbelievable", "wrong", "nobody",
    },
    "learning": {
        "learn", "lesson", "tip", "guide", "explain", "solution",
        "method", "strategy", "important", "remember", "understand",
        "reason", "example", "answer", "because", "how", "why",
    },
    "emotion": {
        "amazing", "incredible", "exciting", "powerful", "success",
        "failure", "problem", "challenge", "best", "worst", "love",
        "hate", "funny", "beautiful", "terrible", "shocked",
    },
    "action": {
        "start", "stop", "try", "use", "make", "build", "create",
        "avoid", "improve", "change", "help", "focus", "practice",
        "work", "choose", "decide", "plan", "solve", "keep",
    },
}

FILLER_WORDS = set("""
a an and are as at be but by for from had has have he her here him
his i if in into is it its just me my of on or our she so that the
their them there they this to us was we were what when where which
who will with you your um uh like yeah okay ok hello hey welcome
""".split())

WEAK_OPENINGS = {
    "hello", "hi", "hey", "welcome", "okay", "ok", "so",
    "anyway", "yeah", "yep", "right", "well",
}


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
        r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)", value
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

    raise ValueError("Unsupported timestamp")


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t-–—|:;,")


def parse_transcript(transcript: str) -> tuple[list[dict[str, Any]], int]:
    entries = []
    skipped = 0

    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue

        matches = list(TIMESTAMP_PATTERN.finditer(line))
        if not matches:
            if entries:
                continuation = clean_text(line)
                if continuation:
                    entries[-1]["text"] += " " + continuation
            else:
                skipped += 1
            continue

        found = False
        for index, match in enumerate(matches):
            try:
                timestamp = parse_timestamp(match.group("timestamp"))
            except ValueError:
                skipped += 1
                continue

            end_pos = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(line)
            )
            text = clean_text(line[match.end():end_pos])
            if not text:
                skipped += 1
                continue

            entries.append({
                "timestamp": round(timestamp, 3),
                "text": text,
            })
            found = True

        if not found:
            skipped += 1

    entries.sort(key=lambda item: item["timestamp"])
    merged = []

    for entry in entries:
        if merged and entry["timestamp"] == merged[-1]["timestamp"]:
            merged[-1]["text"] += " " + entry["text"]
        else:
            merged.append(entry.copy())

    return merged, skipped


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD_PATTERN.findall(text)]


def meaningful_words(text: str) -> list[str]:
    return [
        word for word in words(text)
        if word not in FILLER_WORDS and len(word) > 2
    ]


def score_text(text: str) -> float:
    all_words = words(text)
    if not all_words:
        return 0.0

    unique = set(all_words)
    score = 0.0

    for group in KEYWORDS.values():
        score += min(2, len(unique.intersection(group))) * 1.6

    score += min(len(meaningful_words(text)), 30) * 0.10

    if len(all_words) < 5:
        score -= 1.5
    if "?" in text:
        score += 1.0
    if "!" in text:
        score += 0.7
    if re.search(
        r"\b(but|however|because|instead|although|except)\b",
        text, re.I
    ):
        score += 0.8
    if re.search(
        r"\b(the reason|the truth|what happened|here's why|"
        r"the problem is|the biggest|the key is|the lesson is)\b",
        text, re.I
    ):
        score += 1.3
    if text.rstrip().endswith((".", "!", "?", "…")):
        score += 0.3
    if all_words and all_words[0] in WEAK_OPENINGS:
        score -= 0.5

    return round(max(score, 0.0), 3)


def make_title(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" .,!?:;—-")
    if not text:
        return "Potential highlight"
    title = " ".join(text.split()[:9])
    if len(text.split()) > 9:
        title += "…"
    if len(title) > 70:
        title = title[:67].rstrip() + "..."
    return title[0].upper() + title[1:]


def explain_score(text: str) -> str:
    tokens = set(words(text))
    reasons = []

    if tokens.intersection(KEYWORDS["hook"]):
        reasons.append("possible attention-grabbing hook")
    if tokens.intersection(KEYWORDS["learning"]):
        reasons.append("lesson or explanation")
    if tokens.intersection(KEYWORDS["emotion"]):
        reasons.append("expressive language")
    if tokens.intersection(KEYWORDS["action"]):
        reasons.append("actionable advice")
    if "?" in text:
        reasons.append("question that may invite curiosity")
    if not reasons:
        reasons.append("content selected from the transcript")

    return "; ".join(reasons).capitalize() + "."


def estimate_duration(entries: list[dict[str, Any]]) -> float:
    last = entries[-1]
    speech_seconds = len(WORD_PATTERN.findall(last["text"])) / 2.5
    return last["timestamp"] + min(60.0, max(4.0, speech_seconds))


def build_candidates(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not entries:
        return []

    duration_total = estimate_duration(entries)
    candidates = []
    seen = set()

    for index, entry in enumerate(entries):
        start = float(entry["timestamp"])

        for target_length in CLIP_LENGTHS:
            end = min(start + target_length, duration_total)
            duration = end - start
            if duration < MIN_CLIP_SECONDS:
                continue

            parts = []
            for item in entries[index:]:
                if item["timestamp"] >= end:
                    break
                if item["text"]:
                    parts.append(item["text"])

            combined = clean_text(" ".join(parts))
            count = len(WORD_PATTERN.findall(combined))
            if count < 8:
                continue

            signature = (round(start, 2), round(end, 2))
            if signature in seen:
                continue
            seen.add(signature)

            meaningful_count = len(meaningful_words(combined))
            density = meaningful_count / max(count, 1)
            length_bonus = min(count, 65) * 0.035
            if count < 15:
                length_bonus -= 1.0
            if re.search(r"[.!?…][\"')\]]?\s*$", combined):
                length_bonus += 0.5

            total = score_text(combined) * 0.7 + length_bonus + density * 2

            candidates.append({
                "start_seconds": round(start, 2),
                "end_seconds": round(end, 2),
                "duration_seconds": round(duration, 2),
                "text": combined,
                "score": round(max(total, 0.0), 3),
                "word_count": count,
            })

    return candidates


def overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    overlap = max(
        0.0,
        min(a["end_seconds"], b["end_seconds"])
        - max(a["start_seconds"], b["start_seconds"]),
    )
    shorter = min(a["duration_seconds"], b["duration_seconds"])
    return overlap / shorter if shorter > 0 else 0.0


def choose_highlights(
    entries: list[dict[str, Any]], limit: int = MAX_HIGHLIGHTS
) -> list[dict[str, Any]]:
    ranked = sorted(
        build_candidates(entries),
        key=lambda item: (item["score"], item["word_count"]),
        reverse=True,
    )
    selected = []

    for candidate in ranked:
        if len(selected) >= limit:
            break
        if candidate["score"] <= 0:
            continue
        if any(overlap_ratio(candidate, item) > 0.35 for item in selected):
            continue
        selected.append(candidate)

    if len(selected) < limit:
        for candidate in ranked:
            if len(selected) >= limit:
                break
            if candidate in selected:
                continue
            if any(
                overlap_ratio(candidate, item) > 0.70
                for item in selected
            ):
                continue
            selected.append(candidate)

    selected.sort(key=lambda item: item["start_seconds"])
    results = []

    for item in selected:
        start = item["start_seconds"]
        end = item["end_seconds"]
        results.append({
            "title": make_title(item["text"]),
            "start_seconds": start,
            "end_seconds": end,
            "timestamp": format_timestamp(start),
            "end_timestamp": format_timestamp(end),
            "duration_seconds": item["duration_seconds"],
            "score": item["score"],
            "reason": explain_score(item["text"]),
            "excerpt": item["text"],
        })

    return results


def extract_video_id(url: str) -> str:
    value = url.strip()
    parsed = urlparse(value if "://" in value else "https://" + value)
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
            if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
                video_id = parts[1]

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise HTTPException(
            status_code=400,
            detail="Please enter a valid YouTube video URL.",
        )
    return video_id


def highlights_response(
    transcript: str, mode: str, video_id: str | None = None
) -> dict[str, Any]:
    entries, skipped = parse_transcript(transcript)
    highlights = choose_highlights(entries)

    if not highlights:
        raise HTTPException(
            status_code=400,
            detail=(
                "No usable timestamped segments were found. Paste a "
                "transcript with timestamps and enough text for clips."
            ),
        )

    result = {
        "status": "success",
        "mode": mode,
        "count": len(highlights),
        "highlights": highlights,
        "skipped_lines": skipped,
        "transcript_segments": len(entries),
        "message": (
            "Suggestions use local scoring rules, not verified virality "
            "predictions. Confirm every timestamp against the source video."
        ),
    }
    if video_id:
        result["video_id"] = video_id
    return result


@app.get("/")
def home() -> dict[str, Any]:
    return {
        "name": "ClipForge AI API",
        "version": APP_VERSION,
        "status": "online",
        "endpoints": [
            "/health",
            "/transcript-highlights",
            "/youtube-transcript-highlights",
            "/transcript-insights",
            "/creator-tools",
            "/video-edit-plan",
            "/docs",
        ],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "version": APP_VERSION,
        "mode": "free local transcript analysis",
    }


@app.get("/ai-test")
def ai_test() -> dict[str, Any]:
    return {
        "status": "success",
        "mode": "local",
        "message": "Local tools are available; no paid AI API is required.",
    }


@app.post("/transcript-highlights")
def transcript_highlights(request: TranscriptRequest) -> dict[str, Any]:
    return highlights_response(
        request.transcript.strip(),
        "local timestamp-aware transcript scoring",
    )


@app.post("/youtube-transcript-highlights")
def youtube_transcript_highlights(
    request: YouTubeRequest,
) -> dict[str, Any]:
    video_id = extract_video_id(request.url)

    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        lines = []
        for snippet in fetched:
            text = re.sub(r"\s+", " ", str(snippet.text)).strip()
            if text:
                lines.append(f"{format_timestamp(float(snippet.start))} {text}")
    except Exception as exc:
        name = type(exc).__name__
        if name in {"TranscriptsDisabled", "NoTranscriptFound"}:
            message = (
                "No accessible English captions were found. Try another "
                "video or paste the transcript manually."
            )
        elif name in {"VideoUnavailable", "InvalidVideoId"}:
            message = "The video is unavailable or the URL is invalid."
        else:
            message = (
                "YouTube captions could not be retrieved. Captions may be "
                "disabled or automated access may be blocked. Paste the "
                "transcript manually instead."
            )
        raise HTTPException(status_code=400, detail=message) from None

    transcript = "\n".join(lines)
    if len(transcript) < 20:
        raise HTTPException(status_code=400, detail="Transcript is empty or too short.")

    return highlights_response(
        transcript,
        "YouTube transcript + local scoring",
        video_id,
    )


@app.post("/transcript-insights")
def transcript_insights(request: TranscriptRequest) -> dict[str, Any]:
    transcript = request.transcript.strip()
    entries, skipped = parse_transcript(transcript)
    if not entries:
        raise HTTPException(
            status_code=400,
            detail="Add timestamped transcript lines to analyze topics.",
        )

    tokens = meaningful_words(" ".join(item["text"] for item in entries))
    common = Counter(tokens).most_common(15)
    quotes = sorted(
        entries,
        key=lambda item: (
            score_text(item["text"]),
            len(item["text"]),
        ),
        reverse=True,
    )[:8]

    return {
        "status": "success",
        "segment_count": len(entries),
        "skipped_lines": skipped,
        "keywords": [{"word": word, "count": count} for word, count in common],
        "memorable_quotes": [
            {
                "timestamp": format_timestamp(item["timestamp"]),
                "text": item["text"],
                "score": score_text(item["text"]),
            }
            for item in quotes
        ],
        "note": "Keyword and quote selection is rule-based, not human judgment.",
    }


@app.post("/video-edit-plan")
def video_edit_plan(request: EditPlanRequest) -> dict[str, Any]:
    if request.end_seconds <= request.start_seconds:
        raise HTTPException(
            status_code=400,
            detail="End time must be later than start time.",
        )

    duration = request.end_seconds - request.start_seconds
    if duration > 180:
        raise HTTPException(
            status_code=400,
            detail="Choose a clip of 180 seconds or less.",
        )

    if request.aspect_ratio not in {"9:16", "16:9", "1:1"}:
        raise HTTPException(
            status_code=400,
            detail="Aspect ratio must be 9:16, 16:9, or 1:1.",
        )

    return {
        "status": "success",
        "plan_only": True,
        "start_seconds": request.start_seconds,
        "end_seconds": request.end_seconds,
        "start_timestamp": format_timestamp(request.start_seconds),
        "end_timestamp": format_timestamp(request.end_seconds),
        "duration_seconds": round(duration, 2),
        "aspect_ratio": request.aspect_ratio,
        "captions": request.captions,
        "crop_to_vertical": request.crop_to_vertical,
        "export_ready": False,
        "message": (
            "Editing plan created. This endpoint does not render a video. "
            "MP4 export needs an uploaded source file and FFmpeg processing."
        ),
    }


@app.post("/creator-tools")
def creator_tools(request: CreatorRequest) -> dict[str, Any]:
    transcript = re.sub(r"\s+", " ", request.transcript).strip()
    if len(transcript) < 10:
        raise HTTPException(status_code=400, detail="Transcript is too short.")

    entries, _ = parse_transcript(transcript)
    source_texts = [item["text"] for item in entries] or [transcript]
    best = sorted(
        source_texts,
        key=lambda text: (score_text(text), len(text)),
        reverse=True,
    )[:3]

    title_seed = make_title(best[0])
    hashtags = []
    for word, _count in Counter(meaningful_words(transcript)).most_common(8):
        tag = re.sub(r"[^a-zA-Z0-9]", "", word)
        if len(tag) >= 4:
            hashtags.append("#" + tag.capitalize())

    hashtags = list(dict.fromkeys(
        ["#Shorts", "#ClipForge"] + hashtags
    ))[:10]

    return {
        "status": "success",
        "platform": request.platform,
        "titles": [
            title_seed,
            f"What You Need to Know About {title_seed[:45]}",
            f"This Changed the Conversation: {title_seed[:40]}",
        ],
        "description": (
            f"Watch this moment and share your thoughts!\n\n"
            f"Created with ClipForge AI.\n\n"
            + " ".join(hashtags)
        ),
        "hashtags": hashtags,
        "thumbnail_text": title_seed[:42],
        "quote_ideas": best,
        "note": (
            "Generated from transcript text using local rules. Review "
            "titles and hashtags before publishing."
        ),
    }

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi


APP_VERSION = "1.3.0"
MAX_HIGHLIGHTS = 15
MIN_CLIP_SECONDS = 10
MAX_CLIP_SECONDS = 45
CLIP_LENGTHS = (15, 25, 35, 45)

app = FastAPI(
    title="ClipForge AI API",
    description=(
        "Timestamp-aware transcript highlight analysis and YouTube "
        "transcript importing using local scoring rules."
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
        "finally", "revealed", "discover", "imagine", "actually",
        "never", "always", "honestly", "crazy", "wild",
        "controversial", "unbelievable", "wrong",
    },
    "learning": {
        "learn", "lesson", "tip", "guide", "explain",
        "solution", "method", "strategy", "step", "steps",
        "important", "remember", "understand", "because",
        "reason", "example", "realize", "meaning", "answer",
    },
    "emotion": {
        "amazing", "incredible", "exciting", "powerful",
        "changed", "success", "fail", "failure", "problem",
        "challenge", "best", "worst", "favorite", "love",
        "hate", "fear", "funny", "beautiful", "terrible",
        "surprised", "believe", "shocked",
    },
    "action": {
        "start", "stop", "try", "use", "make", "build",
        "create", "avoid", "improve", "change", "help",
        "focus", "practice", "work", "choose", "decide",
        "plan", "solve", "keep", "protect",
    },
}

FILLER_WORDS = set(
    """
    a an and are as at be but by for from had has have he her
    here him his i if in into is it its just me my of on or our
    she so that the their them there they this to us was we were
    what when where which who will with you your um uh like
    """
    .split()
)

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


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t-–—|:;,")


def parse_transcript(
    transcript: str,
) -> tuple[list[dict[str, Any]], int]:
    entries: list[dict[str, Any]] = []
    skipped_lines = 0

    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue

        matches = list(TIMESTAMP_PATTERN.finditer(line))

        if not matches:
            # Preserve untimestamped continuation lines.
            if entries:
                continuation = clean_text(line)
                if continuation:
                    entries[-1]["text"] += " " + continuation
            else:
                skipped_lines += 1
            continue

        found = 0

        for index, match in enumerate(matches):
            try:
                timestamp = parse_timestamp(match.group("timestamp"))
            except (ValueError, OverflowError):
                skipped_lines += 1
                continue

            end_position = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(line)
            )

            text = clean_text(line[match.end():end_position])

            if not text:
                skipped_lines += 1
                continue

            entries.append({
                "timestamp": round(timestamp, 3),
                "text": text,
            })
            found += 1

        if not found:
            skipped_lines += 1

    entries.sort(key=lambda item: item["timestamp"])

    merged: list[dict[str, Any]] = []

    for entry in entries:
        if merged and entry["timestamp"] == merged[-1]["timestamp"]:
            merged[-1]["text"] += " " + entry["text"]
        else:
            merged.append(entry.copy())

    return merged, skipped_lines


def meaningful_words(text: str) -> list[str]:
    return [
        word.lower()
        for word in WORD_PATTERN.findall(text)
        if word.lower() not in FILLER_WORDS and len(word) > 2
    ]


def score_entry(text: str) -> float:
    words = [word.lower() for word in WORD_PATTERN.findall(text)]

    if not words:
        return 0.0

    unique = set(words)
    score = 0.0

    for group in KEYWORD_GROUPS.values():
        score += min(2, len(unique.intersection(group))) * 1.6

    meaningful = meaningful_words(text)
    score += min(len(meaningful), 30) * 0.10

    if len(words) < 5:
        score -= 1.5

    if "?" in text:
        score += 1.0
    if "!" in text:
        score += 0.7
    if re.search(
        r"\b(but|however|because|instead|although|except)\b",
        text,
        re.IGNORECASE,
    ):
        score += 0.8
    if re.search(
        r"\b(the reason|the truth|what happened|here's why|"
        r"the problem is|the biggest|the key is|the lesson is)\b",
        text,
        re.IGNORECASE,
    ):
        score += 1.3
    if text.rstrip().endswith((".", "!", "?", "…")):
        score += 0.3

    first_word = words[0] if words else ""
    if first_word in WEAK_OPENINGS:
        score -= 0.5

    return round(max(score, 0.0), 3)


def make_title(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" .,!?:;—-")

    if not text:
        return "Potential highlight"

    words = text.split()
    title = " ".join(words[:9])

    if len(words) > 9:
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
        reasons.append("contains a lesson or explanation")
    if words.intersection(KEYWORD_GROUPS["emotion"]):
        reasons.append("uses expressive or impactful language")
    if words.intersection(KEYWORD_GROUPS["action"]):
        reasons.append("contains actionable advice")
    if "?" in text:
        reasons.append("uses a question")
    if re.search(
        r"\b(but|however|because|instead|although)\b",
        text,
        re.IGNORECASE,
    ):
        reasons.append("includes a contrast or explanation")

    if not reasons:
        reasons.append("selected from the available transcript")

    return "; ".join(reasons).capitalize() + "."


def estimate_transcript_duration(
    entries: list[dict[str, Any]],
) -> float:
    last = entries[-1]
    word_count = len(WORD_PATTERN.findall(last["text"]))
    estimated_speech = word_count / 2.5

    # This is an estimate only, not a verified video duration.
    tail = min(60.0, max(4.0, estimated_speech))

    return last["timestamp"] + tail


def build_candidates(
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not entries:
        return []

    video_duration = estimate_transcript_duration(entries)
    candidates = []
    seen = set()

    for start_index, entry in enumerate(entries):
        start = float(entry["timestamp"])

        for target_length in CLIP_LENGTHS:
            end = min(start + target_length, video_duration)
            duration = end - start

            if duration < MIN_CLIP_SECONDS:
                continue

            segment_texts = []

            for item in entries[start_index:]:
                if item["timestamp"] >= end:
                    break
                if item["text"]:
                    segment_texts.append(item["text"])

            combined_text = clean_text(" ".join(segment_texts))
            words = WORD_PATTERN.findall(combined_text)

            if len(words) < 8:
                continue

            # Avoid generating the same window repeatedly.
            signature = (round(start, 2), round(end, 2))
            if signature in seen:
                continue
            seen.add(signature)

            base_score = score_entry(combined_text)
            word_count = len(words)
            meaningful_count = len(meaningful_words(combined_text))

            # Prefer useful, coherent segments without rewarding
            # length alone. This is a heuristic, not an AI prediction.
            density = meaningful_count / max(word_count, 1)
            length_bonus = min(word_count, 65) * 0.035
            density_bonus = density * 2.0

            if word_count < 15:
                length_bonus -= 1.0

            # A complete thought is often easier to use as a short.
            if re.search(r"[.!?…][\"')\]]?\s*$", combined_text):
                length_bonus += 0.5

            total_score = (
                base_score * 0.7
                + length_bonus
                + density_bonus
            )

            candidates.append({
                "start_seconds": round(start, 2),
                "end_seconds": round(end, 2),
                "duration_seconds": round(duration, 2),
                "text": combined_text,
                "score": round(max(total_score, 0.0), 3),
                "word_count": word_count,
            })

    return candidates


def overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
    overlap = max(
        0.0,
        min(a["end_seconds"], b["end_seconds"])
        - max(a["start_seconds"], b["start_seconds"]),
    )

    shorter = min(
        a["duration_seconds"],
        b["duration_seconds"],
    )

    return overlap / shorter if shorter > 0 else 0.0


def choose_highlights(
    entries: list[dict[str, Any]],
    limit: int = MAX_HIGHLIGHTS,
) -> list[dict[str, Any]]:
    candidates = build_candidates(entries)

    if not candidates:
        return []

    ranked = sorted(
        candidates,
        key=lambda item: (
            item["score"],
            item["word_count"],
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []

    # First pass: prefer strong clips that do not repeat the same moment.
    for candidate in ranked:
        if len(selected) >= limit:
            break

        if candidate["score"] <= 0:
            continue

        if any(
            overlap_ratio(candidate, chosen) > 0.35
            for chosen in selected
        ):
            continue

        selected.append(candidate)

    # Second pass: allow additional moments if fewer than requested
    # were found. Short transcripts may not contain 15 good clips.
    if len(selected) < limit:
        for candidate in ranked:
            if len(selected) >= limit:
                break

            if candidate in selected:
                continue

            if any(
                overlap_ratio(candidate, chosen) > 0.70
                for chosen in selected
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
                "Check that your transcript contains timestamps and enough "
                "text to form clips."
            ),
        )

    response = {
        "status": "success",
        "mode": mode,
        "count": len(highlights),
        "highlights": highlights,
        "skipped_lines": skipped,
        "transcript_segments": len(entries),
        "message": (
            "Timestamp-aware highlight suggestions generated using local "
            "heuristics. Scores are not verified virality predictions, "
            "and timestamps have not been checked against the actual video."
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
        "mode": "local timestamp-aware transcript scoring",
    }


@app.post("/transcript-highlights")
def transcript_highlights(
    request: TranscriptRequest,
) -> dict[str, Any]:
    return get_highlights_response(
        request.transcript.strip(),
        "local timestamp-aware transcript scoring",
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
        "YouTube transcript + local timestamp-aware scoring",
        video_id,
)

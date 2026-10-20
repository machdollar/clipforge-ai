import re
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


APP_VERSION = "1.1.0"

app = FastAPI(
    title="ClipForge AI API",
    description=(
        "Finds potential highlights in timestamped transcripts. "
        "This version uses local text-scoring rules and does not "
        "make paid AI API calls or create video files."
    ),
    version=APP_VERSION,
)

# Allow the GitHub Pages frontend to call this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://machdollar.github.io",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class TranscriptRequest(BaseModel):
    transcript: str = Field(
        ...,
        min_length=20,
        max_length=100000,
        description="A timestamped video transcript.",
    )


# Supported timestamp formats include:
# 0:45
# 01:02:15
# 75s
# 80.5 seconds
# 1m 20s
# 1min 20s
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

TIMESTAMP_INLINE_PATTERN = re.compile(
    r"(?<!\S)(?:[\[(]\s*)?"
    r"(?P<timestamp>" + TIMESTAMP_TOKEN + r")"
    r"\s*[\])}]?\s*(?:[-–—|]\s*)?",
    re.IGNORECASE,
)

WORD_PATTERN = re.compile(r"[a-zA-Z0-9']+")

# Words and phrases that can indicate useful highlight moments.
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

FILLER_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but",
    "by", "for", "from", "had", "has", "have", "he",
    "her", "here", "him", "his", "i", "if", "in",
    "into", "is", "it", "its", "just", "me", "my",
    "of", "on", "or", "our", "she", "so", "that",
    "the", "their", "them", "there", "they", "this",
    "to", "us", "was", "we", "were", "what", "when",
    "where", "which", "who", "will", "with", "you",
    "your",
}


def parse_timestamp(value: str) -> float:
    """Convert a supported timestamp into seconds."""
    value = value.strip().lower()
    value = value.strip("[](){} ")

    # Hours:minutes:seconds, such as 01:02:15
    if re.fullmatch(r"\d{1,3}:\d{2}:\d{2}(?:\.\d+)?", value):
        hours, minutes, seconds = value.split(":")
        result = (
            int(hours) * 3600
            + int(minutes) * 60
            + float(seconds)
        )
        if int(minutes) >= 60 or float(seconds) >= 60:
            raise ValueError("Invalid timestamp.")
        return result

    # Minutes:seconds, such as 1:25
    if re.fullmatch(r"\d{1,4}:\d{2}(?:\.\d+)?", value):
        minutes, seconds = value.split(":")
        if float(seconds) >= 60:
            raise ValueError("Invalid timestamp.")
        return int(minutes) * 60 + float(seconds)

    # Seconds, such as 75s or 80.5 seconds
    seconds_match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)",
        value,
    )
    if seconds_match:
        return float(seconds_match.group(1))

    # Minutes with optional seconds, such as 1m 20s or 1min 20
    minutes_match = re.fullmatch(
        r"(\d+)\s*m(?:in(?:ute)?s?)?"
        r"(?:\s*(\d+(?:\.\d+)?)\s*s?)?",
        value,
    )
    if minutes_match:
        minutes = int(minutes_match.group(1))
        seconds = float(minutes_match.group(2) or 0)
        if seconds >= 60:
            raise ValueError("Invalid timestamp.")
        return minutes * 60 + seconds

    raise ValueError("Unsupported timestamp format.")


def parse_transcript(transcript: str) -> tuple[list[dict[str, Any]], int]:
    """
    Parse a transcript containing timestamps.

    Supports multiple timestamps on one line and timestamps on
    separate lines. Text before the first timestamp is ignored.
    """
    entries: list[dict[str, Any]] = []
    skipped_lines = 0

    for line in transcript.splitlines():
        line = line.strip()

        if not line:
            continue

        matches = list(TIMESTAMP_INLINE_PATTERN.finditer(line))

        if not matches:
            skipped_lines += 1
            continue

        found_on_line = 0

        for index, match in enumerate(matches):
            timestamp_text = match.group("timestamp")

            try:
                seconds = parse_timestamp(timestamp_text)
            except (ValueError, OverflowError):
                skipped_lines += 1
                continue

            content_start = match.end()

            if index + 1 < len(matches):
                content_end = matches[index + 1].start()
            else:
                content_end = len(line)

            content = line[content_start:content_end].strip()
            content = content.strip(" -–—|:;,.")
            content = re.sub(r"\s+", " ", content)

            if not content:
                skipped_lines += 1
                continue

            entries.append(
                {
                    "timestamp": round(seconds, 2),
                    "text": content,
                }
            )
            found_on_line += 1

        if found_on_line == 0:
            skipped_lines += 1

    # Sort chronologically.
    entries.sort(key=lambda item: item["timestamp"])

    # Merge duplicate timestamps rather than returning duplicate cards.
    merged: list[dict[str, Any]] = []

    for entry in entries:
        if (
            merged
            and merged[-1]["timestamp"] == entry["timestamp"]
        ):
            merged[-1]["text"] += " " + entry["text"]
        else:
            merged.append(entry.copy())

    return merged, skipped_lines


def score_entry(text: str) -> float:
    """Estimate how interesting a transcript segment may be."""
    words = [
        word.lower()
        for word in WORD_PATTERN.findall(text)
    ]

    if not words:
        return 0.0

    score = 0.0
    unique_words = set(words)

    for group in KEYWORD_GROUPS.values():
        score += len(unique_words.intersection(group)) * 1.5

    # Give some value to meaningful content words.
    meaningful_words = [
        word for word in words
        if word not in FILLER_WORDS and len(word) > 2
    ]
    score += min(len(meaningful_words), 20) * 0.12

    # Very short segments often lack enough context.
    if len(words) < 4:
        score -= 1.0

    # Questions can make useful hooks.
    if "?" in text:
        score += 0.8

    # Emphasis may indicate an expressive moment.
    if "!" in text:
        score += 0.5

    # Prefer complete, readable segments.
    if text.rstrip().endswith((".", "!", "?", "…")):
        score += 0.4

    return round(max(score, 0.0), 3)


def make_title(text: str) -> str:
    """Create a short title from the transcript segment."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.strip(" .,!?:;—-")

    if not cleaned:
        return "Potential highlight"

    words = cleaned.split()
    title_words = words[:8]
    title = " ".join(title_words)

    if len(words) > 8:
        title += "…"

    if len(title) > 70:
        title = title[:67].rstrip() + "..."

    return title[0].upper() + title[1:] if title else "Potential highlight"


def choose_highlights(
    entries: list[dict[str, Any]],
    limit: int = 15,
) -> list[dict[str, Any]]:
    """Rank transcript entries and select varied highlight suggestions."""
    if not entries:
        return []

    scored_entries = []

    for index, entry in enumerate(entries):
        text = entry["text"]
        score = score_entry(text)

        scored_entries.append(
            {
                "index": index,
                "timestamp": entry["timestamp"],
                "text": text,
                "score": score,
            }
        )

    # Higher-scoring entries are considered first.
    ranked = sorted(
        scored_entries,
        key=lambda item: (
            item["score"],
            len(item["text"]),
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    selected_timestamps: set[float] = set()

    # First pass: choose strong entries with some time between them.
    for candidate in ranked:
        if len(selected) >= limit:
            break

        if candidate["score"] <= 0:
            continue

        timestamp = candidate["timestamp"]

        if any(
            abs(timestamp - chosen["timestamp"]) < 12
            for chosen in selected
        ):
            continue

        selected.append(candidate)
        selected_timestamps.add(timestamp)

    # Second pass: fill remaining places with other entries.
    # This makes the endpoint useful for shorter transcripts too.
    if len(selected) < limit:
        for candidate in ranked:
            if len(selected) >= limit:
                break

            timestamp = candidate["timestamp"]

            if timestamp in selected_timestamps:
                continue

            selected.append(candidate)
            selected_timestamps.add(timestamp)

    # Return the selected suggestions in chronological order.
    selected.sort(key=lambda item: item["timestamp"])

    results: list[dict[str, Any]] = []

    for index, item in enumerate(selected):
        timestamp = item["timestamp"]

        if index + 1 < len(selected):
            next_timestamp = selected[index + 1]["timestamp"]
            suggested_end = max(timestamp + 5, next_timestamp)
        else:
            suggested_end = timestamp + 30

        results.append(
            {
                "title": make_title(item["text"]),
                "start_seconds": round(timestamp, 2),
                "end_seconds": round(suggested_end, 2),
                "timestamp": format_timestamp(timestamp),
                "score": item["score"],
                "reason": explain_score(item["text"]),
                "excerpt": item["text"],
            }
        )

    return results


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60

    if hours:
        return (
            f"{hours:02d}:{minutes:02d}:"
            f"{remaining_seconds:02d}"
        )

    return f"{minutes:02d}:{remaining_seconds:02d}"


def explain_score(text: str) -> str:
    """Give a simple explanation for the suggested highlight."""
    words = {
        word.lower()
        for word in WORD_PATTERN.findall(text)
    }

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


@app.get("/")
def home() -> dict[str, Any]:
    return {
        "name": "ClipForge AI API",
        "version": APP_VERSION,
        "status": "online",
        "message": (
            "ClipForge AI backend is running. "
            "Use /docs to explore the API."
        ),
        "endpoints": {
            "health": "/health",
            "transcript_highlights": "/transcript-highlights",
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
    transcript = request.transcript.strip()

    if len(transcript) < 20:
        return {
            "status": "error",
            "mode": "local transcript scoring",
            "count": 0,
            "highlights": [],
            "skipped_lines": 0,
            "message": (
                "Please provide a longer transcript "
                "containing timestamps and text."
            ),
        }

    entries, skipped_lines = parse_transcript(transcript)

    if not entries:
        return {
            "status": "error",
            "mode": "local transcript scoring",
            "count": 0,
            "highlights": [],
            "skipped_lines": skipped_lines,
            "message": (
                "No timestamped transcript segments were found. "
                "Try formats such as 0:05, 1:20, 01:02:15, "
                "or 75s, followed by the spoken text."
            ),
        }

    highlights = choose_highlights(entries)

    return {
        "status": "success",
        "mode": "local transcript scoring",
        "count": len(highlights),
        "highlights": highlights,
        "skipped_lines": skipped_lines,
        "message": (
            "Highlight suggestions generated. "
            "These are heuristic suggestions, not verified "
            "viral moments or actual video clips."
        ),
    }
    

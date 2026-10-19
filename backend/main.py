import re
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Application
# ---------------------------------------------------------

app = FastAPI(
    title="ClipForge AI API",
    description="Transcript-based video highlight suggestions",
    version="1.0.0",
)

# Allow the GitHub Pages frontend to call this Render API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://machdollar.github.io",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MAX_HIGHLIGHTS = 15
MAX_TRANSCRIPT_LENGTH = 100_000
MAX_LINE_LENGTH = 5_000
MAX_CLIP_SECONDS = 60.0
DEFAULT_LAST_CLIP_SECONDS = 30.0
MIN_SPACING_SECONDS = 12.0


# ---------------------------------------------------------
# Request model
# ---------------------------------------------------------

class TranscriptRequest(BaseModel):
    transcript: str = Field(
        min_length=20,
        max_length=MAX_TRANSCRIPT_LENGTH,
    )


# ---------------------------------------------------------
# Timestamp parsing
# Supported examples:
# [00:45] Spoken words
# [01:02:15] Spoken words
# 00:45 - Spoken words
# [75s] Spoken words
# [80.5 seconds] Spoken words
# [1m 20s] Spoken words
# [1min 20s] Spoken words
# ---------------------------------------------------------

TIMESTAMP_PATTERN = re.compile(
    r"^\s*[\[(]?\s*"
    r"(?P<timestamp>"
    r"\d{1,3}:\d{2}:\d{2}(?:\.\d+)?"
    r"|"
    r"\d{1,4}:\d{2}(?:\.\d+)?"
    r"|"
    r"\d+(?:\.\d+)?\s*(?:seconds?|secs?|s)"
    r"|"
    r"\d+\s*m(?:in(?:ute)?s?)?"
    r"(?:\s*\d+(?:\.\d+)?\s*s?)?"
    r")"
    r"\s*[\])}]?\s*"
    r"(?P<text>.*)$",
    re.IGNORECASE,
)

WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)

SIGNAL_GROUPS = {
    "lesson": {
        "important", "lesson", "learn", "remember",
        "understand", "realize", "realise", "insight",
        "wisdom", "advice",
    },
    "mistake": {
        "mistake", "wrong", "failure", "fail", "avoid",
        "warning", "danger", "risk", "problem", "issue",
    },
    "solution": {
        "solution", "solve", "fix", "improve", "method",
        "strategy", "technique", "step", "tip", "trick",
        "guide",
    },
    "surprise": {
        "secret", "truth", "surprising", "unexpected",
        "shocking", "amazing", "incredible", "actually",
        "revealed",
    },
    "emphasis": {
        "key", "best", "worst", "always", "never", "must",
        "essential", "critical", "powerful", "difference",
    },
    "story": {
        "happened", "story", "experience", "finally",
        "result", "changed", "discovered", "realized",
        "realised",
    },
}

FILLER_WORDS = {
    "a", "an", "the", "and", "or", "but", "so", "because",
    "to", "of", "in", "on", "at", "for", "with", "from",
    "is", "are", "was", "were", "be", "been", "being",
    "it", "this", "that", "these", "those", "i", "you",
    "we", "they", "he", "she", "my", "your", "our",
    "me", "us", "them", "do", "does", "did", "have",
    "has", "had", "can", "could", "will", "would",
    "just", "very", "really", "there", "here",
}


def parse_timestamp(value: str) -> float:
    """Convert a supported timestamp format to seconds."""

    value = value.strip().lower()

    # Formats such as 1m20s, 1min 20s, and 2minutes.
    if re.search(r"[a-z]", value):
        minute_match = re.fullmatch(
            r"(\d+)\s*m(?:in(?:ute)?s?)?"
            r"(?:\s*(\d+(?:\.\d+)?)\s*s?)?",
            value,
        )

        if minute_match:
            minutes = int(minute_match.group(1))
            seconds = float(minute_match.group(2) or 0)

            if seconds >= 60:
                raise ValueError(
                    "Seconds in a minute-based timestamp must be below 60."
                )

            return minutes * 60 + seconds

        seconds_match = re.fullmatch(
            r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)",
            value,
        )

        if seconds_match:
            return float(seconds_match.group(1))

        raise ValueError("Unsupported timestamp format.")

    parts = value.split(":")

    try:
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])

            if seconds >= 60:
                raise ValueError("Seconds must be below 60.")

            return minutes * 60 + seconds

        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])

            if minutes >= 60 or seconds >= 60:
                raise ValueError(
                    "Minutes and seconds must be below 60."
                )

            return hours * 3600 + minutes * 60 + seconds

    except ValueError as exc:
        raise ValueError("Invalid timestamp.") from exc

    raise ValueError("Unsupported timestamp format.")


# ---------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------

def parse_transcript(text: str) -> tuple[list[dict[str, Any]], int]:
    """Parse timestamped lines, merge duplicate times, and sort."""

    entries: list[dict[str, Any]] = []
    skipped_lines = 0

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()

        if not line:
            continue

        match = TIMESTAMP_PATTERN.match(line)

        if not match:
            skipped_lines += 1
            continue

        timestamp_text = match.group("timestamp")
        content = match.group("text").strip()
        content = content.lstrip(" \t-–—|:")

        try:
            seconds = parse_timestamp(timestamp_text)
        except ValueError:
            skipped_lines += 1
            continue

        if not content:
            skipped_lines += 1
            continue

        if seconds < 0:
            skipped_lines += 1
            continue

        entries.append({
            "start_seconds": float(seconds),
            "text": content[:MAX_LINE_LENGTH],
            "line_number": line_number,
        })

    if not entries:
        raise HTTPException(
            status_code=422,
            detail=(
                "No valid timestamped transcript lines were found. "
                "Examples: [00:45] Spoken words, "
                "[01:02:15] Spoken words, or [75s] Spoken words."
            ),
        )

    entries.sort(
        key=lambda item: (
            item["start_seconds"],
            item["line_number"],
        )
    )

    # Combine text when multiple lines share the same timestamp.
    merged: list[dict[str, Any]] = []

    for entry in entries:
        if (
            merged
            and entry["start_seconds"] == merged[-1]["start_seconds"]
        ):
            merged[-1]["text"] += " " + entry["text"]
            merged[-1]["text"] = merged[-1]["text"][:MAX_LINE_LENGTH]
        else:
            merged.append({
                "start_seconds": entry["start_seconds"],
                "text": entry["text"],
            })

    return merged, skipped_lines


# ---------------------------------------------------------
# Heuristic scoring
# No external AI API is used.
# ---------------------------------------------------------

def score_entry(text: str) -> dict[str, Any]:
    """Score a transcript line using exact word matches."""

    words = [word.lower() for word in WORD_PATTERN.findall(text)]
    word_count = len(words)
    unique_count = len(set(words))

    groups_found: list[str] = []
    signal_words: list[str] = []

    for group, terms in SIGNAL_GROUPS.items():
        hits = sorted(set(words).intersection(terms))

        if hits:
            groups_found.append(group)
            signal_words.extend(hits)

    score = 0.0
    reasons: list[str] = []

    # A signal category adds value, with a modest bonus for
    # multiple distinct terms in that category.
    for group in groups_found:
        hits = set(words).intersection(SIGNAL_GROUPS[group])
        score += min(2.0, 1.25 + 0.25 * max(0, len(hits) - 1))
        reasons.append(group)

    if "?" in text:
        score += 1.25
        reasons.append("question")

    if "!" in text:
        score += 0.5
        reasons.append("strong emphasis")

    if word_count >= 8:
        score += 0.75

    if word_count >= 15:
        score += 0.5

    if word_count and unique_count / word_count >= 0.65:
        score += 0.25

    # Penalize extremely short fragments.
    if word_count < 4:
        score -= 1.5

    # Penalize artificial-looking repeated characters.
    if re.search(r"(.)\1{4,}", text):
        score -= 0.5

    if not reasons:
        reasons.append("general content")

    return {
        "score": round(max(0.0, score), 2),
        "signals": reasons,
        "signal_words": signal_words,
        "word_count": word_count,
    }


# ---------------------------------------------------------
# Meaningful titles
# ---------------------------------------------------------

def make_title(text: str) -> str:
    """Create a descriptive title based on exact words and punctuation."""

    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"^[\W_]+|[\W_]+$", "", cleaned)

    words = [word.lower() for word in WORD_PATTERN.findall(cleaned)]
    word_set = set(words)

    if "?" in cleaned:
        prefix = "The Big Question"
    elif word_set.intersection(SIGNAL_GROUPS["mistake"]):
        prefix = "A Common Mistake"
    elif word_set.intersection(SIGNAL_GROUPS["solution"]):
        prefix = "A Useful Tip"
    elif word_set.intersection(SIGNAL_GROUPS["surprise"]):
        prefix = "A Surprising Point"
    elif word_set.intersection(SIGNAL_GROUPS["lesson"]):
        prefix = "The Key Lesson"
    elif word_set.intersection(SIGNAL_GROUPS["story"]):
        prefix = "A Story Worth Hearing"
    else:
        prefix = "Key Moment"

    # Use meaningful words from the excerpt, excluding common filler.
    title_words = [
        word for word in WORD_PATTERN.findall(cleaned)
        if word.lower() not in FILLER_WORDS
    ]

    if not title_words:
        title_words = WORD_PATTERN.findall(cleaned)

    excerpt = " ".join(title_words[:6]).strip(" ,;:-")

    if len(title_words) > 6:
        excerpt += "…"

    if not excerpt:
        return prefix

    return f"{prefix}: {excerpt}"[:120]


# ---------------------------------------------------------
# Highlight selection
# ---------------------------------------------------------

def choose_highlights(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank transcript entries and return suggestions in time order."""

    candidates: list[dict[str, Any]] = []

    for index, entry in enumerate(entries):
        metrics = score_entry(entry["text"])

        candidates.append({
            **entry,
            **metrics,
            "original_index": index,
        })

    # Higher-scoring lines rank first; earlier lines break ties.
    ranked = sorted(
        candidates,
        key=lambda item: (
            -item["score"],
            item["start_seconds"],
            item["original_index"],
        ),
    )

    selected: list[dict[str, Any]] = []

    # Prefer variety across the timeline.
    for candidate in ranked:
        if len(selected) >= MAX_HIGHLIGHTS:
            break

        too_close = any(
            abs(
                candidate["start_seconds"] - chosen["start_seconds"]
            ) < MIN_SPACING_SECONDS
            for chosen in selected
        )

        if too_close:
            continue

        selected.append(candidate)

    # If spacing leaves too few results, fill remaining slots with
    # the highest-ranked entries not already selected.
    selected_ids = {
        item["original_index"] for item in selected
    }

    for candidate in ranked:
        if len(selected) >= min(MAX_HIGHLIGHTS, len(candidates)):
            break

        if candidate["original_index"] not in selected_ids:
            selected.append(candidate)
            selected_ids.add(candidate["original_index"])

    selected.sort(key=lambda item: item["start_seconds"])

    highlights: list[dict[str, Any]] = []

    for index, item in enumerate(selected):
        start = item["start_seconds"]

        if index + 1 < len(selected):
            next_start = selected[index + 1]["start_seconds"]
            end = min(next_start, start + MAX_CLIP_SECONDS)
        else:
            end = start + DEFAULT_LAST_CLIP_SECONDS

        if end <= start:
            end = start + DEFAULT_LAST_CLIP_SECONDS

        signal_text = ", ".join(item["signals"])

        highlights.append({
            "start_seconds": round(start, 2),
            "end_seconds": round(end, 2),
            "title": make_title(item["text"]),
            "reason": (
                f"Rank score {item['score']:.2f}. "
                f"Text signals: {signal_text}. "
                "This is a heuristic suggestion, not a verified "
                "clip boundary. Review the video before editing."
            ),
            "transcript_excerpt": item["text"],
        })

    return highlights


# ---------------------------------------------------------
# API routes
# ---------------------------------------------------------

@app.get("/")
def home():
    return {
        "name": "ClipForge AI API",
        "status": "online",
        "version": "1.0.0",
        "mode": "no-api-credit",
        "endpoints": [
            "/",
            "/health",
            "/transcript-highlights",
            "/docs",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "no-api-credit",
    }


@app.post("/transcript-highlights")
def transcript_highlights(request: TranscriptRequest):
    entries, skipped_lines = parse_transcript(request.transcript)
    highlights = choose_highlights(entries)

    return {
        "status": "success",
        "mode": "no-api-credit",
        "count": len(highlights),
        "highlights": highlights,
        "skipped_lines": skipped_lines,
        "message": (
            "Highlights were ranked using local text heuristics, "
            "without an external AI API. These are suggestions, "
            "not verified clip boundaries. Check timestamps "
            "before editing."
        ),
}
            

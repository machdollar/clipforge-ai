from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://machdollar.github.io",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)
import re
from collections import Counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="ClipForge AI API",
    description="Free transcript-based video highlight suggestions",
    version="0.7.0",
)

MAX_HIGHLIGHTS = 15
MAX_LINE_LENGTH = 5000
MAX_CLIP_SECONDS = 60.0
DEFAULT_LAST_CLIP_SECONDS = 30.0
MIN_SPACING_SECONDS = 12.0


class TranscriptRequest(BaseModel):
    transcript: str = Field(min_length=20, max_length=100000)


# Supports:
# [00:10] Text
# [01:20] Text
# [00:01:20] Text
# 00:01:20 - Text
# [80.5s] Text
# [1m 20s] Text
TIMESTAMP_PATTERN = re.compile(
    r"^\s*[\[(]?\s*"
    r"(?P<timestamp>"
    r"(?:\d{1,3}:)?\d{1,2}:\d{2}(?:\.\d+)?"
    r"|"
    r"\d+(?:\.\d+)?\s*(?:seconds?|secs?|s)"
    r"|"
    r"\d+\s*m(?:in(?:ute)?s?)?"
    r"(?:\s*\d+(?:\.\d+)?\s*s?)?"
    r")"
    r"\s*[\]) ]?\s*"
    r"(?P<text>.*)$",
    re.IGNORECASE,
)

WORD_PATTERN = re.compile(r"\b[\w'-]+\b", re.UNICODE)

SIGNAL_GROUPS = {
    "lesson": {
        "important", "lesson", "learn", "remember", "understand",
        "realize", "realise", "insight", "wisdom", "advice",
    },
    "mistake": {
        "mistake", "wrong", "failure", "fail", "avoid", "warning",
        "danger", "risk", "problem", "issue",
    },
    "solution": {
        "solution", "solve", "fix", "improve", "method", "strategy",
        "technique", "step", "tip", "trick", "guide",
    },
    "surprise": {
        "secret", "truth", "surprising", "unexpected", "shocking",
        "amazing", "incredible", "actually", "revealed",
    },
    "emphasis": {
        "key", "best", "worst", "always", "never", "must",
        "essential", "critical", "powerful", "difference",
    },
    "story": {
        "happened", "story", "experience", "finally", "because",
        "result", "changed", "discovered", "realized", "realised",
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
    """Convert a timestamp into seconds."""

    value = value.strip().lower()

    # Examples: 1m20s, 1min 20s, 90 seconds, 80.5s
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
                raise ValueError("Seconds must be below 60.")
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
                raise ValueError("Invalid minutes or seconds.")
            return hours * 3600 + minutes * 60 + seconds

    except ValueError as exc:
        raise ValueError("Invalid timestamp.") from exc

    raise ValueError("Unsupported timestamp format.")


def parse_transcript(text: str) -> list:
    """Read timestamped lines, merge duplicate timestamps, and sort."""

    entries = []
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
                "Examples: [00:45] Spoken words, [01:02:15] Spoken words, "
                "or [75s] Spoken words."
            ),
        )

    entries.sort(key=lambda item: (item["start_seconds"], item["line_number"]))

    # Merge entries sharing the same timestamp.
    merged = []
    for entry in entries:
        if (
            merged
            and entry["start_seconds"] == merged[-1]["start_seconds"]
        ):
            merged[-1]["text"] += " " + entry["text"]
        else:
            merged.append({
                "start_seconds": entry["start_seconds"],
                "text": entry["text"],
            })

    return merged, skipped_lines


def score_entry(text: str) -> dict:
    """Calculate a transparent heuristic score; no AI API is called."""

    lower = text.lower()
    words = WORD_PATTERN.findall(lower)
    word_count = len(words)
    unique_words = len(set(words))

    group_hits = {}
    for group, terms in SIGNAL_GROUPS.items():
        group_hits[group] = sorted({
            word for word in words if word in terms
        })

    score = 0.0
    reasons = []

    for group, hits in group_hits.items():
        if hits:
            score += min(2.5, 1.25 + 0.25 * (len(hits) - 1))
            reasons.append(group)

    if "?" in text:
        score += 1.5
        reasons.append("question")

    if "!" in text:
        score += 0.5
        reasons.append("strong emphasis")

    if word_count >= 8:
        score += 0.75
    if word_count >= 15:
        score += 0.5

    if word_count and unique_words / word_count >= 0.65:
        score += 0.5

    # Very short fragments are less useful on their own.
    if word_count < 4:
        score -= 1.5

    # Repeated letters or punctuation should not dominate ranking.
    if re.search(r"(.)\1{4,}", text):
        score -= 0.5

    if not reasons:
        reasons.append("substantive transcript line")

    return {
        "score": round(max(0.0, score), 2),
        "signals": reasons,
        "word_count": word_count,
    }


def make_title(text: str, signals: list) -> str:
    """Create a short title from the actual transcript excerpt."""

    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"^[\W_]+|[\W_]+$", "", cleaned)

    lower = cleaned.lower()

    if "?" in cleaned:
        prefix = "The Big Question"
    elif any(word in lower for word in SIGNAL_GROUPS["mistake"]):
        prefix = "A Common Mistake"
    elif any(word in lower for word in SIGNAL_GROUPS["solution"]):
        prefix = "A Useful Tip"
    elif any(word in lower for word in SIGNAL_GROUPS["surprise"]):
        prefix = "A Surprising Point"
    elif any(word in lower for word in SIGNAL_GROUPS["lesson"]):
        prefix = "The Key Lesson"
    elif any(word in lower for word in SIGNAL_GROUPS["story"]):
        prefix = "A Story Worth Hearing"
    else:
        prefix = "Key Moment"

    words = cleaned.split()
    excerpt = " ".join(words[:7]).strip(" ,;:-")

    if len(words) > 7:
        excerpt += "…"

    if not excerpt:
        return prefix

    title = f"{prefix}: {excerpt}"
    return title[:120]


def choose_highlights(entries: list) -> list:
    """Rank moments while encouraging variety across the timeline."""

    candidates = []

    for index, entry in enumerate(entries):
        metrics = score_entry(entry["text"])

        candidates.append({
            **entry,
            **metrics,
            "original_index": index,
        })

    # Prefer high-scoring entries, then earlier timestamps.
    candidates.sort(
        key=lambda item: (
            -item["score"],
            item["start_seconds"],
            item["original_index"],
        )
    )

    selected = []

    for candidate in candidates:
        if len(selected) >= MAX_HIGHLIGHTS:
            break

        # Avoid returning nearly identical moments too close together.
        too_close = any(
            abs(candidate["start_seconds"] - item["start_seconds"])
            < MIN_SPACING_SECONDS
            for item in selected
        )

        if too_close and candidate["score"] < 3.0:
            continue

        selected.append(candidate)

    # If spacing filtered everything too aggressively, fill remaining slots.
    if len(selected) < min(MAX_HIGHLIGHTS, len(candidates)):
        selected_ids = {item["original_index"] for item in selected}

        for candidate in candidates:
            if len(selected) >= MAX_HIGHLIGHTS:
                break
            if candidate["original_index"] not in selected_ids:
                selected.append(candidate)
                selected_ids.add(candidate["original_index"])

    selected.sort(key=lambda item: item["start_seconds"])

    highlights = []

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
            "start_seconds": start,
            "end_seconds": round(end, 2),
            "title": make_title(item["text"], item["signals"]),
            "reason": (
                f"Rank score {item['score']:.2f}. "
                f"Text signals: {signal_text}. "
                "This is a heuristic suggestion; review the video "
                "to confirm the moment is useful."
            ),
            "transcript_excerpt": item["text"],
        })

    return highlights


@app.get("/")
def home():
    return {
        "name": "ClipForge AI API",
        "status": "online",
        "version": "0.7.0",
        "mode": "no-api-credit",
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
            "without an external AI API. They are suggestions, "
            "not verified clip boundaries. Check timestamps before editing."
        ),
            }
    

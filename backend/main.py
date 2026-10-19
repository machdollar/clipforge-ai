import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="ClipForge AI API",
    description="Transcript-based highlight organizer",
    version="0.6.0",
)


class TranscriptRequest(BaseModel):
    transcript: str = Field(min_length=20, max_length=100000)


def parse_transcript(text: str):
    entries = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        match = re.match(
            r"^\[?(\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?\]?\s*[-:]?\s*(.+)$",
            line,
        )

        if match:
            timestamp_text = match.group(0).split(" ", 1)[0].strip("[]")
            content = match.group(2).strip()

            parts = timestamp_text.split(":")
            try:
                if len(parts) == 3:
                    seconds = (
                        int(parts[0]) * 3600
                        + int(parts[1]) * 60
                        + float(parts[2])
                    )
                elif len(parts) == 2:
                    seconds = int(parts[0]) * 60 + float(parts[1])
                else:
                    continue
            except ValueError:
                continue

            if content:
                entries.append({
                    "start_seconds": float(seconds),
                    "text": content,
                })

    return entries


@app.get("/")
def home():
    return {
        "name": "ClipForge AI API",
        "status": "online",
        "version": "0.6.0",
    }


@app.get("/health")
def health():
    return {"status": "ok", "mode": "no-api-credit"}


@app.post("/transcript-highlights")
def transcript_highlights(request: TranscriptRequest):
    entries = parse_transcript(request.transcript)

    if not entries:
        raise HTTPException(
            status_code=422,
            detail=(
                "No timestamped lines found. Use one line per entry, "
                "for example: [00:01:20] The speaker explains the main idea."
            ),
        )

    # Score lines using simple text signals. This is a heuristic,
    # not a trained AI model and does not understand the full video.
    signals = (
        "important", "secret", "truth", "mistake", "never",
        "always", "why", "how", "learn", "amazing", "best",
        "worst", "problem", "solution", "remember", "because",
        "surprising", "finally", "tip", "key", "lesson",
    )

    scored = []
    for index, entry in enumerate(entries):
        text = entry["text"]
        lower = text.lower()
        words = re.findall(r"\b[\w'-]+\b", text)

        score = sum(1 for signal in signals if signal in lower)
        if "?" in text:
            score += 2
        if len(words) >= 8:
            score += 1
        if len(words) >= 15:
            score += 1

        scored.append({
            **entry,
            "score": score,
            "original_index": index,
        })

    # Prefer interesting lines spread across the transcript.
    scored.sort(key=lambda item: (-item["score"], item["start_seconds"]))
    selected = scored[:15]
    selected.sort(key=lambda item: item["start_seconds"])

    highlights = []
    for number, item in enumerate(selected, start=1):
        highlights.append({
            "start_seconds": item["start_seconds"],
            "end_seconds": (
                selected[number]["start_seconds"]
                if number < len(selected)
                and selected[number]["start_seconds"] > item["start_seconds"]
                else item["start_seconds"] + 30
            ),
            "title": "Highlight " + str(number),
            "reason": (
                "Selected using transcript text signals. "
                "Review this moment to decide whether it is engaging."
            ),
            "transcript_excerpt": item["text"],
        })

    return {
        "status": "success",
        "mode": "no-api-credit",
        "count": len(highlights),
        "highlights": highlights,
        "message": (
            "These are heuristic suggestions, not AI-generated analysis. "
            "Check the timestamps before editing your video."
        ),
                }
    

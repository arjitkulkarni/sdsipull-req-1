"""HTTP service: ``uvicorn meeting_agent.api:app``.

POST /process returns the verified outcome and a draft that is always marked
``awaiting_approval``. There is deliberately no endpoint that sends anything.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import date

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .llm import ClaudeExtractor, ExtractionError, Extractor
from .models import MeetingOutcome
from .pipeline import run
from .transcript import MAX_TRANSCRIPT_CHARS, TranscriptError, parse_transcript
from .verify import Roster

RATE_LIMIT = 10        # requests
RATE_WINDOW = 60.0     # seconds, per client

app = FastAPI(title="Meeting-to-Action Agent", version="1.0.0")
_hits: dict[str, deque[float]] = defaultdict(deque)


class ProcessRequest(BaseModel):
    transcript: str = Field(max_length=MAX_TRANSCRIPT_CHARS)
    attendees: list[str] = Field(min_length=1, max_length=100)
    meeting_date: date
    title: str = Field(default="Meeting", max_length=200)
    redact_pii: bool = True


class ProcessResponse(BaseModel):
    outcome: MeetingOutcome
    draft: str
    status: str = "awaiting_approval"


def get_extractor() -> Extractor:
    return ClaudeExtractor()


def rate_limit(request: Request) -> None:
    key = request.client.host if request.client else "unknown"
    now, window = time.monotonic(), _hits[key]
    while window and now - window[0] > RATE_WINDOW:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        raise HTTPException(429, "rate limit exceeded")
    window.append(now)


@app.post("/process", response_model=ProcessResponse, dependencies=[Depends(rate_limit)])
def process(req: ProcessRequest, extractor: Extractor = Depends(get_extractor)) -> ProcessResponse:
    try:
        t = parse_transcript(req.transcript, req.meeting_date, req.title)
        result = run(t, Roster(req.attendees), extractor, redact_pii=req.redact_pii)
    except TranscriptError as e:
        raise HTTPException(422, str(e)) from e
    except ExtractionError as e:
        raise HTTPException(502, f"extraction failed closed: {e}") from e
    return ProcessResponse(outcome=result.outcome, draft=result.draft)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}

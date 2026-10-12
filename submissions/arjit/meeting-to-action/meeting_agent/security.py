"""Input guardrails: PII redaction and prompt-injection detection.

Detection here is defence in depth, not the main control. The main controls are
structural: the transcript is passed as delimited data, the model can only return
a fixed schema, and the agent has no tools that act on the outside world.
"""
from __future__ import annotations

import hashlib
import re

from .transcript import Segment, Transcript

INJECTION = re.compile(
    r"(ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?)"
    r"|disregard (all |any |the )?(previous|prior|above|system) "
    r"|you are now\b|new instructions?:|system prompt|developer message"
    r"|\b(ai|llm|assistant|chatgpt|claude|bot)\s*(assistant)?\s*[:,]\s*(please )?(ignore|assign|mark|set|change|delete|output|reveal)"
    r"|assign (every|all|each) (task|action|item)s?"
    r"|mark (every|all|each) (task|action|item)s? as)",
    re.I,
)

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
_CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")


def detect_injections(t: Transcript) -> list[Segment]:
    return [s for s in t.segments if INJECTION.search(s.text)]


def redact(t: Transcript) -> tuple[Transcript, int]:
    """Replace emails, phone and card numbers before anything leaves the machine."""
    count = 0

    def sub(text: str) -> str:
        nonlocal count
        for pattern, label in ((_EMAIL, "[EMAIL]"), (_CARD, "[NUMBER]"), (_PHONE, "[PHONE]")):
            text, n = pattern.subn(label, text)
            count += n
        return text

    segs = [Segment(s.index, s.timestamp, s.speaker, sub(s.text)) for s in t.segments]
    return Transcript(segs, t.meeting_date, t.title), count


def fingerprint(text: str) -> str:
    """Content hash for audit logs, so logs never hold meeting content."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]

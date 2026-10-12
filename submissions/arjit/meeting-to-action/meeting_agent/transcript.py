"""Transcript parsing and quote grounding.

Supported inputs:
  * plain text  - ``[00:02:18] Marcus: text``  or  ``Marcus: text``
  * WebVTT      - cue timing line, then ``<v Marcus>text`` or ``Marcus: text``
  * SRT         - index line, timing line, then ``Marcus: text``
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher

MAX_TRANSCRIPT_CHARS = 400_000  # ~2.5 h of speech; larger input is rejected, never truncated


class TranscriptError(ValueError):
    pass


@dataclass(frozen=True)
class Segment:
    index: int
    timestamp: str
    speaker: str
    text: str

    def render(self) -> str:
        return f"[{self.timestamp}] {self.speaker}: {self.text}"


@dataclass
class Transcript:
    segments: list[Segment]
    meeting_date: date
    title: str = "Meeting"
    _norm_cache: dict[int, str] = field(default_factory=dict, repr=False)

    @property
    def text(self) -> str:
        return "\n".join(s.render() for s in self.segments)

    @property
    def speakers(self) -> list[str]:
        return sorted({s.speaker for s in self.segments})

    def _norm(self, seg: Segment) -> str:
        if seg.index not in self._norm_cache:
            self._norm_cache[seg.index] = normalise(seg.text)
        return self._norm_cache[seg.index]

    def find_quote(self, quote: str, timestamp: str | None = None,
                   min_ratio: float = 0.90) -> tuple[Segment | None, float]:
        """Return the segment that best contains ``quote`` and the match ratio.

        An ellipsis in the quote ("I can look at ... at some point") splits it into
        parts that must all appear, in order, in the same segment. The segment at
        ``timestamp`` is preferred when it matches.
        """
        parts = [normalise(p) for p in re.split(r"\.\.\.|…", quote) if normalise(p)]
        if not parts:
            return None, 0.0
        ordered = sorted(self.segments, key=lambda s: s.timestamp != timestamp)
        best: tuple[Segment | None, float] = (None, 0.0)
        for seg in ordered:
            hay = self._norm(seg)
            ratio = min(_contains_ratio(p, hay) for p in parts)
            if ratio > best[1]:
                best = (seg, ratio)
            if ratio == 1.0:
                break
        return best if best[1] >= min_ratio else (None, best[1])


def normalise(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"[^\w\s']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _contains_ratio(needle: str, hay: str) -> float:
    if needle in hay:
        return 1.0
    n_words, h_words = needle.split(), hay.split()
    if not n_words or not h_words:
        return 0.0
    width = len(n_words)
    best = 0.0
    for i in range(max(1, len(h_words) - width + 1)):
        window = " ".join(h_words[i:i + width])
        best = max(best, SequenceMatcher(None, needle, window).ratio())
    return best


# --------------------------------------------------------------------------- parsing
_BRACKET = re.compile(r"^\[(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]\s*(?P<spk>[^:]{1,60}):\s*(?P<txt>.+)$")
_PLAIN = re.compile(r"^(?P<spk>[A-Z][\w .'-]{0,40}):\s*(?P<txt>.+)$")
_CUE = re.compile(r"^(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)[.,]\d{3}\s*-->\s*")
_VOICE = re.compile(r"^<v\s+(?P<spk>[^>]+)>(?P<txt>.*?)(?:</v>)?$")


def parse_transcript(raw: str, meeting_date: date, title: str = "Meeting") -> Transcript:
    if len(raw) > MAX_TRANSCRIPT_CHARS:
        raise TranscriptError(f"transcript is {len(raw):,} chars; limit is {MAX_TRANSCRIPT_CHARS:,}")
    raw = raw.replace("\r\n", "\n").lstrip("﻿")
    segments = _parse_cues(raw) if _looks_like_cues(raw) else _parse_lines(raw)
    if not segments:
        raise TranscriptError("no speaker lines found; expected 'Speaker: text' lines, VTT or SRT")
    return Transcript(segments=segments, meeting_date=meeting_date, title=title)


def _looks_like_cues(raw: str) -> bool:
    return raw.startswith("WEBVTT") or bool(re.search(r"^\d{1,2}:\d{2}:\d{2}[.,]\d{3}\s*-->", raw, re.M))


def _parse_lines(raw: str) -> list[Segment]:
    out: list[Segment] = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _BRACKET.match(line)
        if m:
            out.append(Segment(len(out), _pad(m["ts"]), m["spk"].strip(), m["txt"].strip()))
            continue
        m = _PLAIN.match(line)
        if m:
            out.append(Segment(len(out), f"L{len(out) + 1:04d}", m["spk"].strip(), m["txt"].strip()))
        elif out:  # continuation of the previous speaker's turn
            prev = out[-1]
            out[-1] = Segment(prev.index, prev.timestamp, prev.speaker, f"{prev.text} {line}")
    return out


def _parse_cues(raw: str) -> list[Segment]:
    out: list[Segment] = []
    ts: str | None = None
    for line in raw.split("\n"):
        line = line.strip()
        cue = _CUE.match(line)
        if cue:
            ts = _pad(cue["ts"])
            continue
        if not line or ts is None or line.isdigit() or line == "WEBVTT":
            continue
        m = _VOICE.match(line) or _PLAIN.match(line)
        if m:
            out.append(Segment(len(out), ts, m["spk"].strip(), m["txt"].strip()))
        elif out and out[-1].timestamp == ts:
            prev = out[-1]
            out[-1] = Segment(prev.index, prev.timestamp, prev.speaker, f"{prev.text} {line}")
    return out


def _pad(ts: str) -> str:
    parts = ts.split(":")
    if len(parts) == 2:
        parts.insert(0, "00")
    return ":".join(p.zfill(2) for p in parts)


def chunk(transcript: Transcript, max_chars: int = 60_000, overlap: int = 1) -> list[Transcript]:
    """Split on speaker turns (never mid-sentence) with ``overlap`` turns of context."""
    if len(transcript.text) <= max_chars:
        return [transcript]
    chunks: list[list[Segment]] = [[]]
    size = 0
    for seg in transcript.segments:
        line = len(seg.render()) + 1
        if size + line > max_chars and chunks[-1]:
            chunks.append(chunks[-1][-overlap:] if overlap else [])
            size = sum(len(s.render()) + 1 for s in chunks[-1])
        chunks[-1].append(seg)
        size += line
    return [Transcript(c, transcript.meeting_date, transcript.title) for c in chunks]

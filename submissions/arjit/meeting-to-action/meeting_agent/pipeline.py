"""End-to-end pipeline: ingest -> extract -> verify -> gate -> draft.

Stage 6 (human review and approval) lives in the CLI / API layer. This module
never sends anything anywhere.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .draft import render_followup
from .llm import Extractor, merge
from .models import MeetingOutcome
from .security import detect_injections, fingerprint, redact
from .transcript import Transcript, chunk
from .verify import Roster, verify


@dataclass
class AuditLog:
    """Append-only record of what the agent did. Holds IDs, flags and hashes, never content."""
    entries: list[dict] = field(default_factory=list)

    def add(self, stage: str, event: str, **detail) -> None:
        self.entries.append({"ts": round(time.time(), 3), "stage": stage, "event": event, **detail})

    def to_jsonl(self) -> str:
        return "".join(json.dumps(e, default=str) + "\n" for e in self.entries)


@dataclass
class RunResult:
    outcome: MeetingOutcome
    draft: str
    audit: AuditLog


def run(transcript: Transcript, roster: Roster, extractor: Extractor, *, redact_pii: bool = False) -> RunResult:
    audit = AuditLog()
    audit.add("ingest", "parsed", segments=len(transcript.segments), speakers=len(transcript.speakers),
              content_sha=fingerprint(transcript.text))
    if redact_pii:
        transcript, n = redact(transcript)
        audit.add("ingest", "redacted", replacements=n)

    unknown = [s for s in transcript.speakers if not roster.resolve(s)]
    if unknown:
        audit.add("ingest", "speakers_not_in_roster", count=len(unknown))

    injected = detect_injections(transcript)
    for seg in injected:
        audit.add("guard", "injection_pattern", at=seg.timestamp)

    parts = chunk(transcript)
    audit.add("extract", "start", chunks=len(parts))
    raw = merge([extractor.extract(p, roster.names) for p in parts])
    audit.add("extract", "done", decisions=len(raw.decisions), action_items=len(raw.action_items),
              model_reported_injections=len(raw.suspected_injections))

    outcome = verify(raw, transcript, roster, injected)
    for a in outcome.action_items:
        audit.add("gate", "action_item", id=a.id, status=a.status, confidence=a.confidence,
                  flags=[f.value for f in a.flags])
    for d in outcome.decisions:
        audit.add("gate", "decision", id=d.id, status=d.status, flags=[f.value for f in d.flags])

    draft = render_followup(outcome)
    audit.add("draft", "rendered", awaiting_approval=True)
    return RunResult(outcome, draft, audit)


def review_report(o: MeetingOutcome) -> str:
    """Markdown report for the reviewer: every item next to the line that supports it."""
    out = [f"# {o.title}: review ({o.meeting_date:%a %d %b %Y})", ""]
    out.append(f"**{len(o.confirmed)} confirmed**, **{len(o.needs_review)} need review**, "
               f"{len(o.decisions)} decisions, {len(o.security_events)} security events.")
    out += ["", "## Action items", "", "| ID | Task | Owner | Due | Confidence | Status | Flags | Evidence |",
            "|---|---|---|---|---|---|---|---|"]
    for a in o.action_items:
        owner = a.owner or (" or ".join(a.owner_candidates) + " (?)" if a.owner_candidates else "-")
        due = f"{a.due:%a %d %b}" if a.due else (f'"{a.due_text}" (unresolved)' if a.due_text else "-")
        ev = f'{a.evidence.timestamp} {a.evidence.speaker}: "{a.evidence.quote}"'
        if not a.evidence.verified:
            ev += " **(not found in transcript)**"
        out.append(f"| {a.id} | {a.task} | {owner} | {due} | {a.confidence:.2f} | {a.status} | "
                   f"{', '.join(f.value for f in a.flags) or '-'} | {ev} |")
    out += ["", "## Decisions", ""]
    for d in o.decisions:
        flags = f" ({', '.join(f.value for f in d.flags)})" if d.flags else ""
        out.append(f'- **{d.id}** [{d.status}] {d.summary} ({d.evidence.timestamp} {d.evidence.speaker}: "{d.evidence.quote}"){flags}')
    if o.security_events:
        out += ["", "## Security events", ""]
        for e in o.security_events:
            out.append(f"- {e.kind} at {e.timestamp} ({e.speaker}), detected by {' + '.join(e.detected_by)}. "
                       "Treated as data; nothing was extracted from this line.")
    notes = [(a.id, n) for a in o.action_items for n in a.notes]
    if notes:
        out += ["", "## Reviewer notes", ""] + [f"- {i}: {n}" for i, n in notes]
    return "\n".join(out) + "\n"

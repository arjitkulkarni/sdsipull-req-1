"""Stage 3 + 4: verify every model claim against the transcript, then gate.

Nothing in this module calls a model. Each check is a plain function so it can
be unit-tested and so every "needs review" carries a reason a person can read.
"""
from __future__ import annotations

import re

from .dates import VAGUE, parse_due
from .models import (BLOCKING, CONFIRM_THRESHOLD, PENALTY, ActionItem, Decision, Evidence, Flag,
                     LLMEvidence, LLMExtraction, MeetingOutcome, OpenQuestion, SecurityEvent)
from .transcript import Segment, Transcript, normalise

QUOTE_MIN_RATIO = 0.90

COMMIT = re.compile(
    r"\b(i'll|i will|i'm going to|i am going to|i can own|i'll own|i'll take|i'll handle|"
    r"i'm on it|i've got (it|this)|leave it with me|i'll do|i'll get|i'll write|i'll send|i'll fix)\b")
HEDGE = re.compile(
    r"\b(maybe|might|perhaps|probably|possibly|at some point|if i (have|get) (the )?time|try to|"
    r"i can (look|take a look|check|have a look)|i could|we'll figure|someone)\b")
ACCEPT = re.compile(r"^(sure|yes|yep|yeah,? (i'll|will do)|will do|on it|ok(ay)?,? (i'll|will do)|i'll|i will|agreed)\b")
REVERSAL = re.compile(
    r"\b(cancel|scratch that|never mind|nevermind|no longer|don't need|do not need|drop (it|that)|"
    r"not needed|instead|reassign|hand (it|that) (over|to))\b")
STOP = {"the", "and", "for", "with", "that", "this", "from", "about", "into", "will", "have",
        "should", "could", "would", "check", "make", "take", "look", "write", "update"}


class Roster:
    """Attendee list with optional aliases, e.g. {"Priya Shah": ["Priya", "PS"]}."""

    def __init__(self, names: list[str], aliases: dict[str, list[str]] | None = None):
        self.names = [n.strip() for n in names if n.strip()]
        self._index: dict[str, set[str]] = {}
        for name in self.names:
            keys = {name, name.split()[0], *(aliases or {}).get(name, [])}
            for k in keys:
                self._index.setdefault(normalise(k), set()).add(name)

    def resolve(self, mention: str | None) -> list[str]:
        if not mention:
            return []
        return sorted(self._index.get(normalise(mention), set()))


# --------------------------------------------------------------------------- evidence
def ground(ev: LLMEvidence, t: Transcript) -> tuple[Evidence, Segment | None, list[Flag]]:
    seg, ratio = t.find_quote(ev.quote, ev.timestamp, QUOTE_MIN_RATIO)
    flags: list[Flag] = []
    if seg is None:
        flags.append(Flag.QUOTE_NOT_FOUND)
        return Evidence(**ev.model_dump(), verified=False, match_ratio=round(ratio, 3)), None, flags
    if normalise(seg.speaker) != normalise(ev.speaker):
        flags.append(Flag.SPEAKER_MISMATCH)
    # Store the transcript's own timestamp/speaker, not the model's claim.
    return Evidence(timestamp=seg.timestamp, speaker=seg.speaker, quote=ev.quote,
                    verified=True, match_ratio=round(ratio, 3)), seg, flags


# --------------------------------------------------------------------------- ownership
def resolve_owner(mentioned: str | None, candidates: list[str], roster: Roster) -> tuple[str | None, list[str], list[Flag]]:
    resolved_candidates = sorted({n for c in candidates for n in (roster.resolve(c) or [c])})
    if len(resolved_candidates) > 1:
        return None, resolved_candidates, [Flag.OWNER_AMBIGUOUS]
    if not mentioned:
        if len(resolved_candidates) == 1 and resolved_candidates[0] in roster.names:
            return resolved_candidates[0], [], []
        return None, resolved_candidates, [Flag.OWNER_MISSING]
    matches = roster.resolve(mentioned)
    if len(matches) == 1:
        return matches[0], [], []
    if len(matches) > 1:
        return None, matches, [Flag.OWNER_AMBIGUOUS]
    return None, [mentioned], [Flag.OWNER_NOT_IN_ROSTER]


def commitment_flags(owner: str | None, seg: Segment | None, t: Transcript, roster: Roster) -> tuple[list[Flag], list[str]]:
    """Strong only if the owner said a commitment phrase themselves, or accepted an assignment."""
    if seg is None or owner is None:
        return [], []
    text = normalise(seg.text)
    speaker_is_owner = owner in roster.resolve(seg.speaker) or normalise(seg.speaker) == normalise(owner)
    if speaker_is_owner:
        if HEDGE.search(text) or not COMMIT.search(text):
            return [Flag.WEAK_COMMITMENT], ["owner hedged or did not use a commitment phrase"]
        return [], []
    # Assigned by someone else: look for the owner accepting in the next few turns.
    for later in t.segments[seg.index + 1: seg.index + 4]:
        if owner in roster.resolve(later.speaker) and ACCEPT.search(normalise(later.text)):
            return [], [f"assigned by {seg.speaker}, accepted by {owner} at {later.timestamp}"]
    return [Flag.WEAK_COMMITMENT], [f"assigned by {seg.speaker}; no acceptance from {owner} found"]


def conflict_flags(task: str, seg: Segment | None, t: Transcript) -> tuple[list[Flag], list[str]]:
    if seg is None:
        return [], []
    keywords = {w for w in normalise(task).split() if len(w) > 3 and w not in STOP}
    for later in t.segments[seg.index + 1:]:
        words = set(normalise(later.text).split())
        if len(keywords & words) >= 2 and REVERSAL.search(normalise(later.text)):
            return [Flag.CONFLICTING_STATEMENTS], [f"possibly reversed at {later.timestamp} by {later.speaker}"]
    return [], []


def due_flags(due_text: str | None, due) -> list[Flag]:
    if not due_text:
        return [Flag.NO_DUE_DATE]
    if due is None or VAGUE.search(due_text.lower()):
        return [Flag.DATE_VAGUE]
    return []


def score(flags: list[Flag]) -> tuple[float, str]:
    f = set(flags)
    if f & {Flag.QUOTE_NOT_FOUND, Flag.INJECTED_CONTENT}:
        return 0.0, "needs_review"  # no trustworthy evidence at all
    conf = max(0.0, 1.0 - sum(PENALTY.get(x, 0.0) for x in f))
    if Flag.CONFLICTING_STATEMENTS in f:
        conf = min(conf, 0.5)
    status = "confirmed" if conf >= CONFIRM_THRESHOLD and not (f & BLOCKING) else "needs_review"
    return round(conf, 2), status


# --------------------------------------------------------------------------- orchestration
def security_events(raw: LLMExtraction, t: Transcript, injected: list[Segment]) -> list[SecurityEvent]:
    """Merge pattern-detected and model-reported injection lines (model reports are grounded first)."""
    events: dict[str, SecurityEvent] = {}
    for s in injected:
        events[s.timestamp] = SecurityEvent(kind="INJECTION_ATTEMPT", timestamp=s.timestamp,
                                            speaker=s.speaker, detected_by=["pattern"])
    for ev in raw.suspected_injections:
        seg, _ = t.find_quote(ev.quote, ev.timestamp, QUOTE_MIN_RATIO)
        if seg is None:
            continue
        if seg.timestamp in events:
            events[seg.timestamp].detected_by.append("model")
        else:
            events[seg.timestamp] = SecurityEvent(kind="INJECTION_ATTEMPT", timestamp=seg.timestamp,
                                                  speaker=seg.speaker, detected_by=["model"])
    return sorted(events.values(), key=lambda e: e.timestamp)


def verify(raw: LLMExtraction, t: Transcript, roster: Roster, injected: list[Segment]) -> MeetingOutcome:
    events = security_events(raw, t, injected)
    injected_ts = {e.timestamp for e in events}

    items: list[ActionItem] = []
    for n, a in enumerate(raw.action_items, 1):
        evidence, seg, flags = ground(a.evidence, t)
        notes: list[str] = []
        if seg is not None and seg.timestamp in injected_ts:
            flags.append(Flag.INJECTED_CONTENT)
            notes.append("evidence comes from a line flagged as a prompt-injection attempt")
        owner, candidates, f = resolve_owner(a.owner_mentioned, a.owner_candidates, roster)
        flags += f
        f, why = commitment_flags(owner, seg, t, roster)
        flags += f
        notes += why
        due = None
        if seg is not None:  # without evidence, a date would only dress up a fabrication
            due = parse_due(a.due_text, t.meeting_date)
            flags += due_flags(a.due_text, due)
        f, why = conflict_flags(a.task, seg, t)
        flags += f
        notes += why
        flags = list(dict.fromkeys(flags))
        conf, status = score(flags)
        items.append(ActionItem(id=f"A{n}", task=a.task, owner=owner, owner_candidates=candidates,
                                due=due if Flag.DATE_VAGUE not in flags else None, due_text=a.due_text,
                                evidence=evidence, flags=flags, notes=notes, confidence=conf, status=status))

    decisions: list[Decision] = []
    for n, d in enumerate(raw.decisions, 1):
        evidence, seg, flags = ground(d.evidence, t)
        if seg is not None and seg.timestamp in injected_ts:
            flags.append(Flag.INJECTED_CONTENT)
        f, _ = conflict_flags(d.summary, seg, t)
        flags += f
        status = "needs_review" if set(flags) & BLOCKING else "confirmed"
        decisions.append(Decision(id=f"D{n}", summary=d.summary, evidence=evidence, flags=flags, status=status))

    questions = []
    for n, q in enumerate(raw.open_questions, 1):
        evidence, seg, _ = ground(q.evidence, t)
        if seg is not None and seg.timestamp not in injected_ts:
            questions.append(OpenQuestion(id=f"Q{n}", question=q.question, evidence=evidence))

    return MeetingOutcome(title=t.title, meeting_date=t.meeting_date, attendees=roster.names,
                          decisions=decisions, action_items=items, open_questions=questions,
                          security_events=events)

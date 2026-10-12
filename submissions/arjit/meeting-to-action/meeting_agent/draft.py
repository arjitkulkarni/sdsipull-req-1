"""Stage 5: the follow-up note.

The draft is rendered from verified data by a template, so it cannot contain a
fact the verifier did not pass. An optional LLM "polish" step can improve tone,
but its output is rejected if it adds, drops or changes any name or date.
"""
from __future__ import annotations

import os
import re

from .models import ActionItem, Flag, MeetingOutcome

REVIEWER_ONLY = frozenset({Flag.QUOTE_NOT_FOUND, Flag.INJECTED_CONTENT})


def _d(day) -> str:
    return f"{day:%a} {day.day} {day:%b}"


def render_followup(o: MeetingOutcome) -> str:
    lines = [f"Subject: {o.title}, {o.meeting_date.day} {o.meeting_date:%b}: decisions & actions", ""]
    lines.append("Hi all, here is the recap from today's meeting.")
    decisions = [d for d in o.decisions if d.status == "confirmed"]
    if decisions:
        lines += ["", "Decisions"]
        lines += [f"  {i}. {d.summary}" for i, d in enumerate(decisions, 1)]
    if o.confirmed:
        lines += ["", "Action items"]
        for a in o.confirmed:
            due = f", due {_d(a.due)}" if a.due else ""
            lines.append(f"  - {a.owner}: {a.task}{due}")
    # Items with no verifiable evidence (or evidence from an injection line) go to the
    # reviewer only. They are never put in front of the team.
    askable = [a for a in o.needs_review if not set(a.flags) & REVIEWER_ONLY]
    if askable:
        lines += ["", "Needs an owner or a date (please reply)"]
        for a in askable:
            lines.append(f"  - {_question(a)}")
    if o.open_questions:
        lines += ["", "Open questions"]
        lines += [f"  - {q.question}" for q in o.open_questions]
    lines += ["", "Reply if anything here is wrong and I'll update it."]
    return "\n".join(lines) + "\n"


def _question(a: ActionItem) -> str:
    f = set(a.flags)
    if Flag.CONFLICTING_STATEMENTS in f:
        return f"{a.task}: this may have been dropped later in the meeting. Is it still needed?"
    if Flag.OWNER_AMBIGUOUS in f:
        return f"{a.task}: {' or '.join(a.owner_candidates)}, who is taking this?"
    if Flag.OWNER_MISSING in f or Flag.OWNER_NOT_IN_ROSTER in f or a.owner is None:
        return f"{a.task}: who owns this?"
    if Flag.DATE_VAGUE in f or Flag.NO_DUE_DATE in f:
        return f"{a.owner}: {a.task}. Can you give a target date?"
    if Flag.WEAK_COMMITMENT in f:
        return f"{a.owner}: {a.task}. Can you confirm you're taking this?"
    return f"{a.owner}: {a.task} (please confirm)"


# --------------------------------------------------------------------------- optional polish
POLISH_PROMPT = """\
Rewrite this meeting follow-up email so it reads naturally. Keep every person's name, every
date and every item exactly as given. Do not add, remove or merge items. Return only the email.

<draft>
{draft}
</draft>"""


def facts(text: str, names: list[str]) -> tuple[set[str], set[str]]:
    """Names and date tokens mentioned in a draft."""
    found = {n for n in names if re.search(rf"\b{re.escape(n.split()[0])}\b", text)}
    dates = set(re.findall(r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\w* \d{1,2} \w{3}", text))
    return found, dates


def polish(draft: str, o: MeetingOutcome, client=None) -> tuple[str, bool]:
    """Return (text, used_polish). Falls back to the template if facts drift."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic(max_retries=3, timeout=120.0)
    resp = client.messages.create(
        model=os.environ.get("MTA_MODEL", "claude-opus-5-5"),
        max_tokens=4000,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": POLISH_PROMPT.format(draft=draft)}],
    )
    if resp.stop_reason != "end_turn":
        return draft, False
    text = "".join(b.text for b in resp.content if b.type == "text").strip() + "\n"
    if facts(text, o.attendees) != facts(draft, o.attendees):
        return draft, False
    return text, True

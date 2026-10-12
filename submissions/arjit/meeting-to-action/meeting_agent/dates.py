"""Deterministic due-date resolution against the meeting date.

Returns ``None`` whenever the phrase does not pin down one calendar day. The
caller turns ``None`` into a DATE_VAGUE flag: we never pick a date for the user.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WD = {d: i for i, d in enumerate(WEEKDAYS)} | {d[:3]: i for i, d in enumerate(WEEKDAYS)}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
_MON = {m: i + 1 for i, m in enumerate(MONTHS)} | {m[:3]: i + 1 for i, m in enumerate(MONTHS)}

# Phrases that sound like a date but are not one.
VAGUE = re.compile(
    r"\b(soon|asap|at some point|sometime|later|eventually|when(ever)? (i|we) can|next sprint|"
    r"next week|this month|next month|in a bit|shortly|before launch|after launch|q[1-4])\b")
ORD = r"(\d{1,2})(?:st|nd|rd|th)?"


def parse_due(text: str | None, meeting: date) -> date | None:
    if not text:
        return None
    t = re.sub(r"\s+", " ", text.lower()).strip()

    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        return _safe(int(m[1]), int(m[2]), int(m[3]))

    # "next friday" is genuinely ambiguous in English (this coming vs. the week after).
    if re.search(r"\bnext (" + "|".join(WEEKDAYS) + r")\b", t):
        return None

    if re.search(r"\b(today|eod|end of (the )?day|tonight)\b", t):
        return meeting
    if re.search(r"\btomorrow\b", t):
        return meeting + timedelta(days=1)
    if re.search(r"\b(eow|end of (the )?week|this week)\b", t):
        return meeting + timedelta(days=(4 - meeting.weekday()) % 7)

    m = re.search(r"\bin (\d{1,2}|a|one|two|three) (day|week)s?\b", t)
    if m:
        n = {"a": 1, "one": 1, "two": 2, "three": 3}.get(m[1]) or int(m[1])
        return meeting + timedelta(days=n * (7 if m[2] == "week" else 1))

    m = re.search(rf"\b({'|'.join(_MON)})\.? {ORD}\b", t) or re.search(rf"\b{ORD} ({'|'.join(_MON)})\b", t)
    if m:
        mon, day = (m[1], m[2]) if m[1].isalpha() else (m[2], m[1])
        d = _safe(meeting.year, _MON[mon], int(day))
        if d and d < meeting:
            d = _safe(meeting.year + 1, _MON[mon], int(day))
        return d

    m = re.search(rf"\bthe {ORD}\b", t)
    if m:
        day = int(m[1])
        d = _safe(meeting.year, meeting.month, day)
        if d is None or d < meeting:
            y, mo = (meeting.year + 1, 1) if meeting.month == 12 else (meeting.year, meeting.month + 1)
            d = _safe(y, mo, day)
        return d

    m = re.search(r"\b(" + "|".join(_WD) + r")\b", t)
    if m and not VAGUE.search(t):
        ahead = (_WD[m[1]] - meeting.weekday()) % 7 or 7  # same weekday means next week
        return meeting + timedelta(days=ahead)

    return None


def _safe(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None

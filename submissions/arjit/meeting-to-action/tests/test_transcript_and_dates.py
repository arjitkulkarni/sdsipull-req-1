from datetime import date

import pytest

from meeting_agent.dates import parse_due
from meeting_agent.transcript import MAX_TRANSCRIPT_CHARS, TranscriptError, chunk, parse_transcript

MON = date(2026, 10, 12)  # a Monday


# --------------------------------------------------------------------------- parsing
def test_bracket_format_with_comments_and_continuations():
    t = parse_transcript("# header\n[00:02:18] Marcus: I'll write it\nand test it.\n[2:30] Sam: ok", MON)
    assert [(s.timestamp, s.speaker) for s in t.segments] == [("00:02:18", "Marcus"), ("00:02:30", "Sam")]
    assert t.segments[0].text == "I'll write it and test it."


def test_vtt_voice_tags():
    raw = "WEBVTT\n\n00:01:40.000 --> 00:01:43.000\n<v Lena>Decision then: second reviewer.\n"
    t = parse_transcript(raw, MON)
    assert (t.segments[0].timestamp, t.segments[0].speaker) == ("00:01:40", "Lena")


def test_srt():
    raw = "1\n00:00:01,000 --> 00:00:03,000\nOmar: Yes, I'll add it.\n\n2\n00:00:04,000 --> 00:00:05,000\nJin: Thanks.\n"
    t = parse_transcript(raw, MON)
    assert [s.speaker for s in t.segments] == ["Omar", "Jin"]


def test_plain_lines_get_line_numbers():
    t = parse_transcript("Priya: hello\nSam: hi", MON)
    assert [s.timestamp for s in t.segments] == ["L0001", "L0002"]


def test_rejects_empty_and_oversized():
    with pytest.raises(TranscriptError):
        parse_transcript("no speakers here", MON)
    with pytest.raises(TranscriptError):
        parse_transcript("A: x\n" * (MAX_TRANSCRIPT_CHARS // 4), MON)


# --------------------------------------------------------------------------- quote grounding
@pytest.fixture
def t():
    return parse_transcript(
        "[00:02:18] Marcus: Agreed. I'll write the rollback runbook by Thursday.\n"
        "[00:04:40] Sam: I can look at the flaky checkout tests at some point.", MON)


def test_exact_quote_found_ignoring_punctuation_and_case(t):
    seg, ratio = t.find_quote("i'll write the rollback runbook, by thursday")
    assert seg.speaker == "Marcus" and ratio == 1.0


def test_near_verbatim_quote_passes_threshold(t):
    seg, ratio = t.find_quote("I'll write the rollback run-book by Thursday")
    assert seg is not None and ratio >= 0.9


def test_ellipsis_quote_needs_every_part(t):
    assert t.find_quote("I can look at ... at some point")[0].speaker == "Sam"
    assert t.find_quote("I can look at ... by Friday")[0] is None


def test_paraphrase_is_rejected(t):
    seg, ratio = t.find_quote("Marcus promised to fix the runbook tomorrow")
    assert seg is None and ratio < 0.9


def test_chunking_keeps_overlap_and_all_segments():
    raw = "\n".join(f"[00:{i // 60:02d}:{i % 60:02d}] S{i % 3}: line number {i} " + "x" * 50 for i in range(200))
    t = parse_transcript(raw, MON)
    parts = chunk(t, max_chars=2000, overlap=1)
    assert len(parts) > 1
    assert parts[1].segments[0] == parts[0].segments[-1]
    assert {s.index for p in parts for s in p.segments} == set(range(200))


# --------------------------------------------------------------------------- dates
@pytest.mark.parametrize("text, expected", [
    ("by Thursday", date(2026, 10, 15)),
    ("on Monday", date(2026, 10, 19)),          # same weekday means next week
    ("tomorrow", date(2026, 10, 13)),
    ("EOD", MON),
    ("by end of week", date(2026, 10, 16)),
    ("the 22nd", date(2026, 10, 22)),
    ("the 3rd", date(2026, 11, 3)),             # already passed this month
    ("Oct 30", date(2026, 10, 30)),
    ("30 October", date(2026, 10, 30)),
    ("Jan 5", date(2027, 1, 5)),                # rolls to next year
    ("2026-11-02", date(2026, 11, 2)),
    ("in 2 weeks", date(2026, 10, 26)),
])
def test_resolvable_dates(text, expected):
    assert parse_due(text, MON) == expected


@pytest.mark.parametrize("text", ["at some point", "soon", "ASAP", "next sprint", "next week",
                                  "next Friday", "before launch", "Feb 30", None, ""])
def test_vague_or_invalid_dates_are_never_guessed(text):
    assert parse_due(text, MON) is None

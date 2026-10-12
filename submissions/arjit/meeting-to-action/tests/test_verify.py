from datetime import date

import pytest

from meeting_agent.models import Flag, LLMActionItem, LLMEvidence, LLMExtraction
from meeting_agent.transcript import parse_transcript
from meeting_agent.verify import Roster, resolve_owner, score, verify

MON = date(2026, 10, 12)
ROSTER = Roster(["Priya Shah", "Marcus Lee", "Dana Cruz", "Sam Ito"], aliases={"Sam Ito": ["Sammy"]})

TRANSCRIPT = parse_transcript("""\
[00:01:00] Marcus Lee: I'll write the rollback runbook by Thursday.
[00:02:00] Priya Shah: Dana, can you update the launch checklist?
[00:02:05] Dana Cruz: Sure, will do by Wednesday.
[00:03:00] Priya Shah: Sam, please draft the load-test plan for the migration.
[00:04:00] Sam Ito: I can look at the flaky tests at some point.
[00:05:00] Priya Shah: Actually, scratch that, drop the load-test plan, platform owns it.
""", MON)


def item(task, owner, ts, speaker, quote, due=None, candidates=()):
    return LLMActionItem(task=task, owner_mentioned=owner, owner_candidates=list(candidates), due_text=due,
                         evidence=LLMEvidence(timestamp=ts, speaker=speaker, quote=quote))


def run_one(i):
    raw = LLMExtraction(decisions=[], action_items=[i], open_questions=[], suspected_injections=[])
    return verify(raw, TRANSCRIPT, ROSTER, injected=[]).action_items[0]


# --------------------------------------------------------------------------- owners
@pytest.mark.parametrize("mention, owner, flags", [
    ("Marcus", "Marcus Lee", []),
    ("marcus lee", "Marcus Lee", []),
    ("Sammy", "Sam Ito", []),
    ("Ravi", None, [Flag.OWNER_NOT_IN_ROSTER]),
    (None, None, [Flag.OWNER_MISSING]),
])
def test_resolve_owner(mention, owner, flags):
    got, _, f = resolve_owner(mention, [], ROSTER)
    assert (got, f) == (owner, flags)


def test_multiple_candidates_is_ambiguous_even_if_model_picked_one():
    got, cands, f = resolve_owner("Dana", ["Dana", "Sam"], ROSTER)
    assert got is None and cands == ["Dana Cruz", "Sam Ito"] and f == [Flag.OWNER_AMBIGUOUS]


def test_first_name_shared_by_two_attendees_is_ambiguous():
    r = Roster(["Sam Ito", "Sam Park"])
    assert resolve_owner("Sam", [], r)[2] == [Flag.OWNER_AMBIGUOUS]


# --------------------------------------------------------------------------- end-to-end item checks
def test_explicit_self_commitment_is_confirmed():
    a = run_one(item("Write rollback runbook", "Marcus", "00:01:00", "Marcus Lee",
                     "I'll write the rollback runbook by Thursday", "by Thursday"))
    assert a.status == "confirmed" and a.owner == "Marcus Lee" and a.due == date(2026, 10, 15)
    assert a.flags == [] and a.confidence == 1.0


def test_assignment_accepted_by_owner_is_confirmed():
    a = run_one(item("Update launch checklist", "Dana", "00:02:00", "Priya Shah",
                     "Dana, can you update the launch checklist?", "by Wednesday"))
    assert a.status == "confirmed" and "accepted by Dana Cruz" in a.notes[0]


def test_hedged_commitment_is_flagged_not_promoted():
    a = run_one(item("Investigate flaky tests", "Sam", "00:04:00", "Sam Ito",
                     "I can look at the flaky tests at some point", "at some point"))
    assert set(a.flags) == {Flag.WEAK_COMMITMENT, Flag.DATE_VAGUE}
    assert a.status == "needs_review" and a.due is None


def test_reversed_item_is_blocked():
    a = run_one(item("Draft the load-test plan for the migration", "Sam", "00:03:00", "Priya Shah",
                     "Sam, please draft the load-test plan for the migration."))
    assert Flag.CONFLICTING_STATEMENTS in a.flags and a.status == "needs_review"
    assert "00:05:00" in " ".join(a.notes)


def test_fabricated_quote_is_blocked_with_zero_confidence():
    a = run_one(item("Book the offsite", "Priya", "00:01:00", "Priya Shah",
                     "Priya will book the offsite for Friday", "Friday"))
    assert a.flags[0] == Flag.QUOTE_NOT_FOUND
    assert a.status == "needs_review" and a.confidence == 0.0 and a.due is None
    assert a.evidence.verified is False


def test_wrong_speaker_attribution_is_corrected_and_flagged():
    a = run_one(item("Write rollback runbook", "Marcus", "00:01:00", "Priya Shah",
                     "I'll write the rollback runbook by Thursday", "by Thursday"))
    assert Flag.SPEAKER_MISMATCH in a.flags
    assert a.evidence.speaker == "Marcus Lee"  # stored from the transcript, not the model


def test_score_never_confirms_with_blocking_flag():
    for f in (Flag.QUOTE_NOT_FOUND, Flag.INJECTED_CONTENT, Flag.CONFLICTING_STATEMENTS):
        assert score([f])[1] == "needs_review"
    assert score([])[1] == "confirmed"
    assert score([Flag.NO_DUE_DATE]) == (0.95, "confirmed")
    assert score([Flag.WEAK_COMMITMENT]) == (0.7, "needs_review")

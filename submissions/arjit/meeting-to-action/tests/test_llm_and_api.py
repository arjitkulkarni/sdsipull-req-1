"""Claude integration (with a fake client: no network, no key) and the HTTP API."""
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from meeting_agent import api
from meeting_agent.draft import polish, render_followup
from meeting_agent.llm import SYSTEM_PROMPT, ClaudeExtractor, ExtractionError, ReplayExtractor, merge
from meeting_agent.models import LLMExtraction
from meeting_agent.transcript import parse_transcript
from meeting_agent.verify import Roster

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
SPRINT_TXT = (SAMPLES / "sprint_planning.txt").read_text(encoding="utf-8")
FIXTURE = ReplayExtractor(SAMPLES / "sprint_planning.extraction.json").extract(None, [])


class FakeMessages:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def create(self, **kwargs):
        return self.parse(**kwargs)


def fake_client(*responses):
    msgs = FakeMessages(responses)
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs), messages=msgs), msgs


def ok(parsed=FIXTURE, stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, parsed_output=parsed)


@pytest.fixture
def transcript():
    return parse_transcript(SPRINT_TXT, date(2026, 10, 12), "Sprint planning")


def test_request_shape(transcript, monkeypatch):
    monkeypatch.setenv("MTA_MODEL", "claude-opus-5-5")
    client, msgs = fake_client(ok())
    out = ClaudeExtractor(client=client).extract(transcript, ["Priya", "Marcus", "Dana", "Sam"])
    assert out == FIXTURE
    call = msgs.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_format"] is LLMExtraction            # schema-constrained output
    assert call["system"] == SYSTEM_PROMPT
    assert "tools" not in call and "tool_choice" not in call  # the model has nothing it can act with
    assert "temperature" not in call                          # not accepted on current models
    content = call["messages"][0]["content"]
    assert "<transcript>\n[00:00:05] Priya:" in content and content.endswith("</transcript>")
    assert "Meeting date: 2026-10-12 (Monday)" in content


def test_system_prompt_treats_transcript_as_data():
    assert "untrusted DATA" in SYSTEM_PROMPT and "Never guess" in SYSTEM_PROMPT


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_fails_closed_on_refusal_or_truncation(transcript, stop):
    client, _ = fake_client(ok(stop=stop))
    with pytest.raises(ExtractionError):
        ClaudeExtractor(client=client).extract(transcript, [])


def test_retries_once_on_schema_error_then_fails_closed(transcript):
    err = ValidationError.from_exception_data("LLMExtraction", [])
    client, msgs = fake_client(err, ok())
    assert ClaudeExtractor(client=client).extract(transcript, []) == FIXTURE
    client, msgs = fake_client(err, err)
    with pytest.raises(ExtractionError):
        ClaudeExtractor(client=client).extract(transcript, [])
    assert len(msgs.calls) == 2


def test_merge_deduplicates_overlapping_chunks():
    merged = merge([FIXTURE, FIXTURE])
    assert len(merged.action_items) == len(FIXTURE.action_items)
    assert len(merged.decisions) == len(FIXTURE.decisions)


# --------------------------------------------------------------------------- polish guard
def _outcome(transcript):
    from meeting_agent.pipeline import run
    return run(transcript, Roster(["Priya", "Marcus", "Dana", "Sam"]), ReplayExtractor(
        SAMPLES / "sprint_planning.extraction.json")).outcome


def _text_resp(text):
    return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])


def test_polish_accepted_when_facts_unchanged(transcript):
    o = _outcome(transcript)
    draft = render_followup(o)
    client, _ = fake_client(_text_resp(draft.replace("Hi all", "Hello team")))
    text, used = polish(draft, o, client=client)
    assert used and "Hello team" in text


def test_polish_rejected_when_it_changes_a_date_or_owner(transcript):
    o = _outcome(transcript)
    draft = render_followup(o)
    for bad in (draft.replace("Thu 15 Oct", "Fri 16 Oct"), draft.replace("Marcus", "Priya")):
        client, _ = fake_client(_text_resp(bad))
        text, used = polish(draft, o, client=client)
        assert not used and text == draft


# --------------------------------------------------------------------------- HTTP API
@pytest.fixture
def http():
    api._hits.clear()
    api.app.dependency_overrides[api.get_extractor] = lambda: ReplayExtractor(
        SAMPLES / "sprint_planning.extraction.json")
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


BODY = {"transcript": SPRINT_TXT, "attendees": ["Priya", "Marcus", "Dana", "Sam"], "meeting_date": "2026-10-12"}


def test_api_process(http):
    r = http.post("/process", json=BODY)
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "awaiting_approval"
    assert [a["status"] for a in j["outcome"]["action_items"]] == ["confirmed", "needs_review", "needs_review", "confirmed"]


def test_api_rejects_bad_input(http):
    assert http.post("/process", json={**BODY, "attendees": []}).status_code == 422
    assert http.post("/process", json={**BODY, "transcript": "nothing to parse"}).status_code == 422
    assert http.post("/process", json={**BODY, "transcript": "x" * 400_001}).status_code == 422


def test_api_rate_limit(http):
    codes = [http.post("/process", json=BODY).status_code for _ in range(api.RATE_LIMIT + 1)]
    assert codes[-1] == 429 and set(codes[:-1]) == {200}


def test_api_has_no_send_endpoint():
    paths = {r.path for r in api.app.routes}
    assert not any(word in p for p in paths for word in ("send", "email", "slack", "post_to"))

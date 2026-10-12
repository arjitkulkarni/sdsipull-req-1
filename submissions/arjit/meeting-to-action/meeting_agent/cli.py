"""Command line entry point.

    python -m meeting_agent run samples/sprint_planning.txt \
        --roster "Priya,Marcus,Dana,Sam" --date 2026-10-12 --title "Sprint planning"

Writes outcome.json, review.md, followup.draft.md and audit.jsonl to --out.
With --review, walks a person through the flagged items and asks for explicit
approval before writing followup.approved.md. Nothing is ever sent.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from .draft import polish, render_followup
from .llm import ClaudeExtractor, ExtractionError, ReplayExtractor
from .models import Flag
from .pipeline import RunResult, review_report, run
from .transcript import TranscriptError, parse_transcript
from .verify import Roster, score


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="meeting_agent", description="Meeting transcript -> verified actions + follow-up draft")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process a transcript")
    r.add_argument("transcript", type=Path)
    r.add_argument("--roster", help="comma-separated attendee names")
    r.add_argument("--roster-file", type=Path, help="one attendee per line")
    r.add_argument("--date", type=date.fromisoformat, required=True, help="meeting date, YYYY-MM-DD")
    r.add_argument("--title", default="Meeting")
    r.add_argument("--out", type=Path, default=Path("out"))
    r.add_argument("--replay", type=Path, help="use a saved extraction JSON instead of calling Claude (offline demo)")
    r.add_argument("--redact", action="store_true", help="redact emails/phone numbers before the model call")
    r.add_argument("--polish", action="store_true", help="let Claude rewrite the draft (fact-checked, falls back to template)")
    r.add_argument("--review", action="store_true", help="interactive human review and approval")
    r.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    names = args.roster.split(",") if args.roster else []
    if args.roster_file:
        names += args.roster_file.read_text(encoding="utf-8").splitlines()
    if not [n for n in names if n.strip()]:
        p.error("an attendee roster is required (--roster or --roster-file); owners are only ever chosen from it")
    roster = Roster(names)

    try:
        transcript = parse_transcript(args.transcript.read_text(encoding="utf-8"), args.date, args.title)
        extractor = ReplayExtractor(args.replay) if args.replay else ClaudeExtractor()
        result = run(transcript, roster, extractor, redact_pii=args.redact)
    except (TranscriptError, ExtractionError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.polish and not args.replay:
        result.draft, used = polish(result.draft, result.outcome)
        result.audit.add("draft", "polish", accepted=used)

    args.out.mkdir(parents=True, exist_ok=True)
    _write(args.out, result)
    _print_summary(result)

    if args.review:
        return _interactive_review(result, args.out)
    print(f"\nDraft written to {args.out / 'followup.draft.md'} and awaiting approval (re-run with --review).")
    return 0


def _write(out: Path, result: RunResult) -> None:
    (out / "outcome.json").write_text(result.outcome.model_dump_json(indent=2), encoding="utf-8")
    (out / "review.md").write_text(review_report(result.outcome), encoding="utf-8")
    (out / "followup.draft.md").write_text("<!-- DRAFT: awaiting human approval -->\n" + result.draft, encoding="utf-8")
    (out / "audit.jsonl").write_text(result.audit.to_jsonl(), encoding="utf-8")


def _print_summary(result: RunResult) -> None:
    o = result.outcome
    print(f"{o.title}: {len(o.decisions)} decisions, {len(o.confirmed)} confirmed actions, "
          f"{len(o.needs_review)} need review, {len(o.security_events)} security events\n")
    for a in o.action_items:
        mark = "OK " if a.status == "confirmed" else "?? "
        owner = a.owner or "/".join(a.owner_candidates) or "-"
        due = f"{a.due:%a %d %b}" if a.due else "-"
        flags = ", ".join(f.value for f in a.flags)
        print(f"  {mark}{a.id:<3} {a.task[:44]:<44} {owner:<12} {due:<11} {a.confidence:.2f}  {flags}")
    for e in o.security_events:
        print(f"  !! {e.kind} at {e.timestamp} ({e.speaker}), ignored")


def _interactive_review(result: RunResult, out: Path) -> int:
    o = result.outcome
    for a in o.needs_review:
        print(f"\n{a.id}: {a.task}\n  flags: {', '.join(f.value for f in a.flags)}")
        print(f'  evidence: {a.evidence.timestamp} {a.evidence.speaker}: "{a.evidence.quote}"')
        choice = input("  [a]ccept as-is / set [o]wner / [r]eject / [s]kip: ").strip().lower()
        if choice == "o":
            name = input(f"  owner ({', '.join(o.attendees)}): ").strip()
            if name in o.attendees:
                a.owner, a.owner_candidates = name, []
                a.flags = [f for f in a.flags if f not in (Flag.OWNER_AMBIGUOUS, Flag.OWNER_MISSING, Flag.OWNER_NOT_IN_ROSTER)]
                choice = "a"
            else:
                print("  not an attendee; skipped")
        if choice == "a" and a.evidence.verified:
            a.status, a.confidence = "confirmed", score(a.flags)[0]
            a.notes.append("confirmed by human reviewer")
        elif choice == "r":
            o.action_items.remove(a)
        result.audit.add("review", "item", id=a.id, choice=choice or "s")

    result.draft = render_followup(o)
    print("\n" + result.draft)
    approved = input("Approve this follow-up? [y/N]: ").strip().lower() == "y"
    result.audit.add("review", "approval", approved=approved)
    _write(out, result)
    if approved:
        (out / "followup.approved.md").write_text(result.draft, encoding="utf-8")
        print(f"Approved draft saved to {out / 'followup.approved.md'}. Send it from your own mail client.")
    else:
        print("Not approved; nothing saved as final.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

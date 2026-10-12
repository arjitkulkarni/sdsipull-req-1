"""Meeting-to-Action Agent: transcript -> verified decisions, action items and a follow-up draft."""
from .pipeline import RunResult, run
from .transcript import parse_transcript
from .verify import Roster

__all__ = ["Roster", "RunResult", "parse_transcript", "run"]

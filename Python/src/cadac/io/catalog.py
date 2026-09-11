from pathlib import Path

from cadac.io.asc_deck import parse_asc_deck
from cadac.io.translate import parse_scenario_asc

SKIP_STEMS = frozenset({"readme", "documentation", "doc", "input_copy"})


def classify_asc(path: Path) -> str:
    path = Path(path)
    if path.stem.lower() in SKIP_STEMS:
        return "skip"
    try:
        payload = parse_scenario_asc(path)
    except Exception:
        payload = None
    if payload is not None and (payload.get("modules") or payload.get("vehicles")):
        return "scenario"
    try:
        _title, tables = parse_asc_deck(path)
    except Exception:
        return "fail"
    return "deck" if tables else "fail"

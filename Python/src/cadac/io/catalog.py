from pathlib import Path

from cadac_cpp.extract_cpp import PROGRAM_DIRS  # noqa: F401

from cadac.io.asc_deck import parse_asc_deck
from cadac.io.translate import (
    deck_asc_to_jsonc,
    parse_scenario_asc,
    translate_scenario_asc,
)

SKIP_STEMS = frozenset({"readme", "documentation", "doc", "input_copy"})

PROGRAM_FAMILY = {
    "AIM5": "aim5", "CRUISE5": "cruise5", "MAGSIX": "magsix",
    "ROCKET6": "rocket6", "SAM6": "sam6", "SRAAM6": "sraam6", "AGM6": "agm6",
}


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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def cases_dir(program: str) -> Path:
    return repo_root() / "Python" / "cases" / program.lower()


def dest_jsonc(program: str, src: Path) -> Path:
    return cases_dir(program) / f"{src.stem}.jsonc"


def family_for(program: str) -> str | None:
    return PROGRAM_FAMILY.get(program)


def translate_one(program: str, src: Path) -> str:
    kind = classify_asc(src)
    if kind == "skip":
        return "skipped"
    if kind == "fail":
        return "failed"
    dst = dest_jsonc(program, src)
    if dst.exists():
        return "exists"
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if kind == "scenario":
            translate_scenario_asc(src, dst.parent, family=family_for(program))
            # translate_scenario_asc writes dst.parent / f"{src.stem}.jsonc"
        else:
            deck_asc_to_jsonc(src, dst)
    except Exception:
        if dst.exists():
            dst.unlink()
        return "failed"
    return "written"

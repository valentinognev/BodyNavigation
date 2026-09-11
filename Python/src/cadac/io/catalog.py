import json
from pathlib import Path

from cadac_cpp.extract_cpp import PROGRAM_DIRS

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

EXTRA_SOURCES: list[tuple[str, str]] = [
    ("HYPER6", "CADAC_Simulations/HYPER6 Input Problems for Sec 10_4"),
    ("AGM6", "CADAC_Simulations/AGM6_250217/Additional input Files"),
]


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


def translate_one(program: str, src: Path) -> tuple[str, str | None]:
    kind = classify_asc(src)
    if kind == "skip":
        return "skipped", None
    if kind == "fail":
        return "failed", "fail"
    dst = dest_jsonc(program, src)
    if dst.exists():
        return "exists", None
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if kind == "scenario":
            translate_scenario_asc(src, dst.parent, family=family_for(program))
            # translate_scenario_asc writes dst.parent / f"{src.stem}.jsonc"
        else:
            deck_asc_to_jsonc(src, dst)
    except Exception as exc:
        if dst.exists():
            dst.unlink()
        return "failed", f"{type(exc).__name__}: {exc}"
    return "written", None


def iter_asc_jobs() -> list[tuple[str, Path]]:
    root = repo_root()
    jobs: list[tuple[str, Path]] = []
    seen: dict[str, set[str]] = {}
    sources: list[tuple[str, str]] = list(PROGRAM_DIRS.items()) + list(EXTRA_SOURCES)
    for program, rel in sources:
        directory = root / rel
        if not directory.is_dir():
            continue
        for src in sorted(directory.glob("*.asc")):
            if src.stem.lower() in SKIP_STEMS:
                continue
            stems = seen.setdefault(program, set())
            if src.stem in stems:
                continue
            stems.add(src.stem)
            jobs.append((program, src))
    return jobs


def _posix_rel(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(repo_root()).as_posix()
    except ValueError:
        return resolved.as_posix()


def run_catalog() -> dict:
    report: dict = {"translated": [], "skipped": [], "failed": []}
    for program, src in iter_asc_jobs():
        status, err = translate_one(program, src)
        dest = dest_jsonc(program, src)
        if status == "written":
            report["translated"].append(_posix_rel(dest))
        elif status in {"exists", "skipped"}:
            report["skipped"].append(_posix_rel(dest))
        elif status == "failed":
            report["failed"].append({"path": _posix_rel(src), "error": err or "failed"})
    write_report(report)
    return report


def write_report(report: dict, path: Path | None = None) -> Path:
    if path is None:
        path = repo_root() / "Python" / "cases" / "catalog-report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path

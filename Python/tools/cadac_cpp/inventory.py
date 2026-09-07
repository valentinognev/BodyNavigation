import re
from pathlib import Path

from cadac_cpp.extract_cpp import (
    PROGRAM_DIRS,
    extract_def_modules,
    extract_modes,
    extract_vehicle_types,
)
from cadac_cpp.extract_python import (
    extract_family_keys,
    extract_global_types,
    extract_python_modes,
)
from cadac_cpp.harvest_table import HARVEST_ROWS, repo_root
from cadac_cpp.kernel_rows import kernel_rows
from cadac_cpp.schema import InventoryRow, dump_inventory

# Task 10 pytest snapshot. Regenerating inventory.json does not re-run e2e.
# Edit this map when e2e results change; do not parse pytest JSON.
_E2E_OUTCOMES = {
    "test_agm6_freeflight.py": "failed",
    "test_agm6_testcase.py": "failed",
    "test_aim5_hori.py": "passed",
    "test_cruise5_input1.py": "passed",
    "test_falcon5_turning.py": "failed",
    "test_falcon6_gamma.py": "passed",
    "test_hyper3_climb.py": "passed",
    "test_hyper5_pronav.py": "failed",
    "test_hyper6_climb.py": "passed",
    "test_magsix_attitude.py": "failed",
    "test_magsix_trajectory.py": "passed",
    "test_rocket6_insertion.py": "failed",
    "test_sam6_autopilot.py": "failed",
    "test_sraam6_1v1.py": "passed",
}
_E2E_NOTES = {
    "test_agm6_freeflight.py": "ValueError: math domain error",
    "test_agm6_testcase.py": "ZeroDivisionError: division by zero",
    "test_falcon5_turning.py": "FSPV3 at t=0.0",
    "test_hyper5_pronav.py": "psivgx at t=0.0",
    "test_magsix_attitude.py": "KeyError: ('sim_time', 0.3501)",
    "test_rocket6_insertion.py": "vmach at t=0.1",
    "test_sam6_autopilot.py": "thtvlcx at t=0.0",
}
_E2E_STATUS = {"passed": "ported", "failed": "diverged", "skipped": "missing"}

_NAME_EQ = re.compile(r'^[ \t]*name[ \t]*=[ \t]*["\']([^"\']+)["\']', re.MULTILINE)
_SLICE_VEHICLES = {("HYPER6", "RADAR0"), ("HYPER6", "SAT3"), ("HYPER6", "Ground0")}


def status_for_mode(cpp_pair, py_implemented, stub_flags) -> str:
    if cpp_pair in py_implemented:
        return "ported"
    if cpp_pair[0] in stub_flags:
        return "stubbed"
    return "missing"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _program_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.cpp")) + sorted(directory.glob("*.hpp"))


def _slice_set_obj_type(text: str) -> str | None:
    start = 0
    key = "set_obj_type"
    while True:
        idx = text.find(key, start)
        if idx < 0:
            return None
        i = idx + len(key)
        while i < len(text) and text[i].isspace():
            i += 1
        if i < len(text) and text[i] == "(":
            depth = 1
            i += 1
            while i < len(text) and depth:
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                i += 1
            while i < len(text) and text[i].isspace():
                i += 1
            if i < len(text) and text[i] == "{":
                depth = 0
                for j, ch in enumerate(text[i:], i):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return text[idx : j + 1]
                return text[idx:]
        start = idx + 1


def _vehicle_text(files: list[Path]) -> tuple[str, Path]:
    by_name = {path.name: path for path in files}
    ordered: list[Path] = []
    preferred = by_name.get("global_functions.cpp")
    if preferred is not None:
        ordered.append(preferred)
    ordered.extend(path for path in files if path != preferred)
    for path in ordered:
        sliced = _slice_set_obj_type(_read_text(path))
        if sliced is not None:
            return sliced, path
    concat = "\n".join(_read_text(path) for path in files)
    return concat, files[0]


def _dotted(src_root: Path, path: Path) -> str:
    return ".".join(path.relative_to(src_root).with_suffix("").parts)


def _python_module_maps(py_files: list[Path], src_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    by_name: dict[str, str] = {}
    by_stem: dict[str, str] = {}
    for path in py_files:
        dotted = _dotted(src_root, path)
        stem = path.stem
        if stem != "__init__":
            by_stem.setdefault(stem, dotted)
        text = _read_text(path)
        for match in _NAME_EQ.finditer(text):
            by_name.setdefault(match.group(1), dotted)
    return by_name, by_stem


def _vehicle_python(name: str, program: str, global_types: set[str], family_keys: set[tuple[str, str]]) -> tuple[str, str | None]:
    family = program.lower()
    if (family, name) in family_keys:
        return "ported", "cadac.cli:_VEHICLE_FAMILIES"
    if name in global_types:
        return "ported", "cadac.cli:_VEHICLE_TYPES"
    return "missing", None


def _python_vehicle_family(python: str | None) -> str | None:
    if not python:
        return None
    parts = python.split(".")
    if len(parts) >= 3 and parts[0] == "cadac" and parts[1] == "vehicles":
        return parts[2]
    return None


def apply_human_notes(rows: list[InventoryRow]) -> list[InventoryRow]:
    annotated: list[InventoryRow] = []
    for row in rows:
        note = row.note
        if row.kind == "vehicle" and row.status in ("missing", "stubbed"):
            if (row.program, row.name) in _SLICE_VEHICLES:
                note = "slice limit"
        elif row.kind == "mode" and row.status in ("missing", "stubbed"):
            flag = row.name.split("=", 1)[0]
            if flag in ("maut", "mauty"):
                note = "slice limit"
            elif row.status == "missing":
                note = "drop-out"
        elif row.kind == "module" and row.status == "ported":
            py_fam = _python_vehicle_family(row.python)
            if py_fam and py_fam != row.program.lower():
                note = "name match only"
        if note != row.note:
            annotated.append(
                InventoryRow(
                    program=row.program,
                    kind=row.kind,
                    name=row.name,
                    cpp=row.cpp,
                    python=row.python,
                    status=row.status,
                    note=note,
                )
            )
        else:
            annotated.append(row)
    return annotated


def _program_from_cpp_dir(cpp_dir: str) -> str:
    for program, rel in PROGRAM_DIRS.items():
        if rel == cpp_dir:
            return program
    raise KeyError(cpp_dir)


def _harvest_dest(root: Path, row) -> Path:
    if row.golden.endswith("hyper3/plot1.csv"):
        return root / "Python/tests/e2e/goldens/hyper3/plot1.gpp.csv"
    return root / row.golden


def harvest_inventory_rows(
    root: Path,
    harvest_results: dict[str, str] | None = None,
    harvest_notes: dict[str, str] | None = None,
) -> list[InventoryRow]:
    notes = harvest_notes or {}
    rows: list[InventoryRow] = []
    for row in HARVEST_ROWS:
        raw = (harvest_results or {}).get(row.golden)
        if raw is None:
            raw = "ok" if _harvest_dest(root, row).is_file() else "missing"
        status = "ported" if raw == "ok" else "missing"
        note = notes.get(row.golden, "")
        rows.append(
            InventoryRow(
                program=_program_from_cpp_dir(row.cpp_dir),
                kind="harvest",
                name=row.golden,
                cpp=f"{row.cpp_dir}/{row.asc_name}",
                python=row.jsonc,
                status=status,
                note=note,
            )
        )
    return rows


def _e2e_jsonc_cases(root: Path) -> list[tuple[Path, object]]:
    e2e_dir = root / "Python" / "tests" / "e2e"
    found: list[tuple[Path, object]] = []
    for path in sorted(e2e_dir.glob("test_*.py")):
        text = _read_text(path)
        matched = [
            row
            for row in HARVEST_ROWS
            if Path(row.jsonc).name in text and Path(row.jsonc).parent.name in text
        ]
        if not matched:
            continue
        found.append((path, matched[0]))
    return found


def e2e_inventory_rows(
    root: Path,
    e2e_outcomes: dict[str, str] | None = None,
    e2e_notes: dict[str, str] | None = None,
) -> list[InventoryRow]:
    outcomes = e2e_outcomes if e2e_outcomes is not None else _E2E_OUTCOMES
    notes = e2e_notes if e2e_notes is not None else _E2E_NOTES
    rows: list[InventoryRow] = []
    for path, row in _e2e_jsonc_cases(root):
        outcome = outcomes.get(path.name, "skipped")
        rows.append(
            InventoryRow(
                program=_program_from_cpp_dir(row.cpp_dir),
                kind="e2e",
                name=row.golden,
                cpp=row.golden,
                python=str(path.relative_to(root)),
                status=_E2E_STATUS.get(outcome, "missing"),
                note=notes.get(path.name, ""),
            )
        )
    return rows


def scan_all(
    root: Path,
    harvest_results: dict[str, str] | None = None,
    harvest_notes: dict[str, str] | None = None,
    e2e_outcomes: dict[str, str] | None = None,
    e2e_notes: dict[str, str] | None = None,
) -> list[InventoryRow]:
    src_root = root / "Python" / "src"
    cadac_root = src_root / "cadac"
    py_files = sorted(p for p in cadac_root.rglob("*.py") if p.is_file())
    cli_text = _read_text(cadac_root / "cli.py")
    py_concat = "\n".join(_read_text(p) for p in py_files)
    global_types = set(extract_global_types(cli_text))
    family_keys = set(extract_family_keys(cli_text))
    implemented, stub_flags = extract_python_modes(py_concat)
    implemented_set = set(implemented)
    by_name, by_stem = _python_module_maps(py_files, src_root)

    rows: list[InventoryRow] = []
    for program, rel in PROGRAM_DIRS.items():
        directory = root / rel
        files = _program_files(directory)
        vehicle_text, vehicle_path = _vehicle_text(files)
        vehicle_cpp = f"{program}/{vehicle_path.name}:set_obj_type"
        for name in extract_vehicle_types(vehicle_text):
            status, python = _vehicle_python(name, program, global_types, family_keys)
            rows.append(
                InventoryRow(
                    program=program,
                    kind="vehicle",
                    name=name,
                    cpp=vehicle_cpp,
                    python=python,
                    status=status,
                    note="",
                )
            )

        modules: dict[str, str] = {}
        modes: dict[tuple[str, int], str] = {}
        for path in files:
            text = _read_text(path)
            cpp = f"{program}/{path.name}"
            for name in extract_def_modules(text):
                modules.setdefault(name, cpp)
            for pair in extract_modes(text):
                modes.setdefault(pair, cpp)

        for name, cpp in modules.items():
            python = by_name.get(name) or by_stem.get(name)
            rows.append(
                InventoryRow(
                    program=program,
                    kind="module",
                    name=name,
                    cpp=cpp,
                    python=python,
                    status="ported" if python else "missing",
                    note="",
                )
            )

        for pair, cpp in modes.items():
            flag, value = pair
            rows.append(
                InventoryRow(
                    program=program,
                    kind="mode",
                    name=f"{flag}={value}",
                    cpp=cpp,
                    python=None,
                    status=status_for_mode(pair, implemented_set, stub_flags),
                    note="",
                )
            )

    rows.extend(kernel_rows())
    rows.extend(harvest_inventory_rows(root, harvest_results, harvest_notes))
    rows.extend(e2e_inventory_rows(root, e2e_outcomes, e2e_notes))
    rows.sort(key=lambda row: (row.program, row.kind, row.name))
    return rows


if __name__ == "__main__":
    dump_inventory(
        Path(__file__).parent / "inventory.json",
        apply_human_notes(scan_all(repo_root())),
    )

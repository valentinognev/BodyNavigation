import json
from pathlib import Path

from cadac.io.asc_deck import parse_asc_deck
from cadac.tables.lookup import Table


def _table_to_dict(table: Table) -> dict:
    raw = {
        "name": table.name,
        "dim": table.dim,
        "x1": table.x1.tolist(),
        "values": table.values.tolist(),
    }
    if table.x2 is not None:
        raw["x2"] = table.x2.tolist()
    if table.x3 is not None:
        raw["x3"] = table.x3.tolist()
    return raw


def deck_asc_to_jsonc(src, dst) -> None:
    title, tables = parse_asc_deck(src)
    payload = {"title": title, "tables": [_table_to_dict(t) for t in tables]}
    Path(dst).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _comment_stripped(line: str) -> str:
    for marker in ("//", "!"):
        idx = line.find(marker)
        if idx >= 0:
            line = line[:idx]
    return line.strip()


def _content_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = _comment_stripped(raw)
        if stripped:
            lines.append(stripped)
    return lines


def _parse_number(token: str):
    digits = token.lstrip("+-")
    if digits.isdigit() and token not in {"+", "-", ""}:
        return int(token)
    return float(token)


_KNOWN_PHASES = frozenset({"def", "init", "exec", "term"})

# Fortran MODULES labels (code + description) → Python module names.
_FTN_MODULE_LABELS = {
    "TARGET": "target",
    "ENVIRON": "environment",
    "ENVIRONMENT": "environment",
    "SENSOR": "seeker",
    "AI RADAR": "ai_radar",
    "INS": "ins",
    "GUIDANCE": "guidance",
    "AUTOPILOT": "control",
    "AERO": "aerodynamics",
    "AERODYNAMICS": "aerodynamics",
    "PROPULSION": "propulsion",
    "FORCES": "forces",
    "NEWTON": "newton",
    "NEWTONS LAW": "newton",
    "ROTATIONS": "rotations",
}

_FTN_EXEC_KEYS = {
    "der": "int_step",
    "ppp": "plot_step",
    "cpp": "scrn_step",
}

_FTN_DEFAULT_PHASES = ["def", "init", "exec"]


def _is_fortran_module_line(mparts: list[str]) -> bool:
    if len(mparts) < 2:
        return False
    phases = [
        phase.strip()
        for phase in "".join(mparts[1:]).split(",")
        if phase.strip()
    ]
    if not phases:
        return False
    return any(phase.lower() not in _KNOWN_PHASES for phase in phases)


def _parse_fortran_module(mparts: list[str]) -> dict:
    label = " ".join(mparts[1:]).upper()
    name = _FTN_MODULE_LABELS.get(label)
    if name is None:
        name = label.lower().replace(" ", "_")
    return {"name": name, "phases": list(_FTN_DEFAULT_PHASES)}


def _parse_fortran_value(raw: str):
    text = raw.strip()
    upper = text.upper()
    if upper.startswith("INT(") and text.endswith(")"):
        return int(float(text[4:-1].strip()))
    if upper.startswith("GAUSS(") and text.endswith(")"):
        inside = text[6:-1]
        mean = inside.split(",")[0].strip()
        return _parse_number(mean)
    return _parse_number(text)


def _parse_fortran_assignment(line: str) -> tuple[str, object] | None:
    if "=" not in line:
        return None
    left, right = line.split("=", 1)
    name = left.strip().lower()
    if not name or name.startswith("func "):
        return None
    # Drop array indices: emisa(1) → emisa kept only if we want; skip arrays for skeleton.
    if "(" in name:
        return None
    return name, _parse_fortran_value(right.strip())


def _parse_fortran_if_when(line: str) -> dict:
    # "IF DBT1 < 1851.97" or "IF TIME > 10.00"
    parts = line.split(None, 1)[1].split()
    name = parts[0].lower()
    if len(parts) >= 3:
        op, raw_value = parts[1], parts[2]
    else:
        token = parts[1]
        op = next((candidate for candidate in _IF_OPS if token.startswith(candidate)), "")
        raw_value = token[len(op) :]
    if op not in _IF_OPS:
        raise ValueError(f"unknown IF operator {op!r}")
    return {name: {op: _parse_number(raw_value)}}


def _family_vehicle_type(family: str | None) -> str:
    if family == "sraam5":
        return "SRAAM5"
    if family == "rocket3":
        return "ROCKET3"
    return "VEHICLE"


def _parse_fortran_body(
    lines: list[str], i: int, *, family: str | None = None
) -> tuple[list[dict], dict[str, float], float, int, int]:
    """Parse CADAC4 Fortran deck body after MODULES END → params/events/timing."""
    params: dict = {}
    events: list[dict] = []
    timing: dict[str, float] = {}
    iseed = 0
    pending_when: dict | None = None
    pending_set: dict = {}
    n = len(lines)

    def flush_event() -> None:
        nonlocal pending_when, pending_set
        if pending_when is not None:
            events.append({"when": pending_when, "set": dict(pending_set)})
            pending_when = None
            pending_set = {}

    while i < n:
        line = lines[i]
        key = line.split()[0].upper()
        if key in {"RUN", "STOP"}:
            flush_event()
            i += 1
            if key == "RUN" and i < n and lines[i].split()[0].upper() == "STOP":
                i += 1
            break
        if key == "IF":
            flush_event()
            pending_when = _parse_fortran_if_when(line)
            i += 1
            continue
        assignment = _parse_fortran_assignment(line)
        if assignment is None:
            i += 1
            continue
        name, value = assignment
        if name == "ranseed":
            iseed = int(value)
            i += 1
            continue
        timing_key = _FTN_EXEC_KEYS.get(name)
        if timing_key is not None:
            timing[timing_key] = float(value)
            i += 1
            continue
        if pending_when is not None:
            pending_set[name] = value
        else:
            params[name] = value
        i += 1

    flush_event()
    vehicle_type = _family_vehicle_type(family)
    vehicle = {
        "type": vehicle_type,
        "name": vehicle_type.lower(),
        "params": params,
        "events": events,
    }
    end_time = 0.0
    return [vehicle], timing, end_time, iseed, i


def _parse_option(token: str) -> tuple[str, bool]:
    if token.startswith("y_"):
        return token[2:], True
    if token.startswith("n_"):
        return token[2:], False
    raise ValueError(f"unknown OPTIONS token {token!r}")


def _deck_jsonc(name: str) -> str:
    return str(Path(name).with_suffix(".jsonc"))


_IF_OPS = frozenset({"<", "=", ">"})


def _parse_if_event(lines: list[str], i: int) -> tuple[dict, int]:
    parts = lines[i].split()
    name = parts[1]
    if len(parts) >= 4:
        op, raw_value = parts[2], parts[3]
    else:
        token = parts[2]
        op = next((candidate for candidate in _IF_OPS if token.startswith(candidate)), "")
        raw_value = token[len(op):]
    if op not in _IF_OPS:
        raise ValueError(f"unknown IF operator {op!r}")
    assignments: dict = {}
    i += 1
    n = len(lines)
    while i < n:
        eparts = lines[i].split()
        if eparts[0] == "ENDIF":
            i += 1
            break
        assignments[eparts[0]] = _parse_number(eparts[1])
        i += 1
    return {"when": {name: {op: _parse_number(raw_value)}}, "set": assignments}, i


def _parse_vehicle(lines: list[str], i: int) -> tuple[dict, int]:
    parts = lines[i].split()
    vehicle_type, name = parts[0], parts[1]
    i += 1
    params: dict = {}
    events: list = []
    aero_deck = None
    prop_deck = None
    weather_deck = None
    sam_deck = None
    srmb_deck = None
    n = len(lines)
    while i < n:
        parts = lines[i].split()
        token = parts[0]
        if token == "END":
            i += 1
            break
        if token == "IF":
            event, i = _parse_if_event(lines, i)
            events.append(event)
            continue
        if token == "ENDIF":
            i += 1
            continue
        if token == "AERO_DECK":
            aero_deck = _deck_jsonc(parts[1])
            i += 1
            continue
        if token == "PROP_DECK":
            prop_deck = _deck_jsonc(parts[1])
            i += 1
            continue
        if token == "WEATHER_DECK":
            weather_deck = _deck_jsonc(parts[1])
            i += 1
            continue
        if token == "SAM_DECK":
            sam_deck = _deck_jsonc(parts[1])
            i += 1
            continue
        if token == "SRBM_DECK":
            srmb_deck = _deck_jsonc(parts[1])
            i += 1
            continue
        if token == "GAUSS":
            params[parts[1]] = _parse_number(parts[2])
            i += 1
            continue
        if token == "RAYL":
            params[parts[1]] = _parse_number(parts[2])
            i += 1
            continue
        if token == "MARKOV":
            params[parts[1]] = 0
            i += 1
            continue
        params[token] = _parse_number(parts[1])
        i += 1
    vehicle: dict = {"type": vehicle_type, "name": name}
    if aero_deck is not None:
        vehicle["aero_deck"] = aero_deck
    if prop_deck is not None:
        vehicle["prop_deck"] = prop_deck
    if weather_deck is not None:
        vehicle["weather_deck"] = weather_deck
    if sam_deck is not None:
        vehicle["sam_deck"] = sam_deck
    if srmb_deck is not None:
        vehicle["srmb_deck"] = srmb_deck
    vehicle["params"] = params
    vehicle["events"] = events
    return vehicle, i


def parse_scenario_asc(src: Path) -> dict:
    lines = _content_lines(src)
    i = 0
    n = len(lines)
    title = ""
    options: dict[str, bool] = {}
    modules: list[dict] = []
    timing: dict[str, float] = {}
    vehicles: list[dict] = []
    end_time = 0.0
    iseed = 0
    nmonte = 0
    fortran_modules = False
    while i < n:
        parts = lines[i].split()
        key = parts[0]
        if key == "TITLE":
            title = lines[i].split(None, 1)[1].strip() if len(parts) > 1 else ""
            i += 1
        elif key == "MONTE":
            # C++ acquire_title_options: MONTE nmonte iseed (no iseed bump between runs)
            nmonte = int(parts[1])
            iseed = int(parts[2])
            i += 1
        elif key == "OPTIONS":
            for token in parts[1:]:
                name, flag = _parse_option(token)
                options[name] = flag
            i += 1
        elif key == "MODULES":
            i += 1
            while i < n and lines[i].split()[0] != "END":
                mparts = lines[i].split()
                if _is_fortran_module_line(mparts):
                    fortran_modules = True
                    modules.append(_parse_fortran_module(mparts))
                else:
                    phases = [
                        phase.strip()
                        for phase in "".join(mparts[1:]).split(",")
                        if phase.strip()
                    ]
                    modules.append({"name": mparts[0], "phases": phases})
                i += 1
            i += 1  # skip END
            if fortran_modules and i < n:
                next_key = lines[i].split()[0].upper()
                if next_key not in {"OPTIONS", "TIMING", "VEHICLES", "ENDTIME"}:
                    vehicles, ftiming, end_time, iseed, i = _parse_fortran_body(
                        lines, i
                    )
                    timing.update(ftiming)
        elif key == "TIMING":
            i += 1
            while i < n and lines[i].split()[0] != "END":
                tparts = lines[i].split()
                timing[tparts[0]] = float(tparts[1])
                i += 1
            i += 1
        elif key == "VEHICLES":
            nveh = int(parts[1])
            i += 1
            for _ in range(nveh):
                vehicle, i = _parse_vehicle(lines, i)
                vehicles.append(vehicle)
            if i < n and lines[i].split()[0] == "END":
                i += 1
        elif key == "ENDTIME":
            end_time = float(parts[1])
            i += 1
        else:
            i += 1
    payload = {
        "title": title,
        "options": options,
        "modules": modules,
        "timing": timing,
        "end_time": end_time,
        "vehicles": vehicles,
    }
    if nmonte:
        payload["nmonte"] = nmonte
    if iseed:
        payload["iseed"] = iseed
    return payload


_parse_scenario_asc = parse_scenario_asc


def translate_scenario_asc(src, dst_dir, family=None) -> None:
    src = Path(src)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    payload = _parse_scenario_asc(src)
    if family is not None:
        payload["family"] = family
        vtype = _family_vehicle_type(family)
        for vehicle in payload["vehicles"]:
            vehicle["family"] = family
            if vehicle.get("type") == "VEHICLE":
                vehicle["type"] = vtype
                vehicle["name"] = vtype.lower()
    dst_dir.joinpath(f"{src.stem}.jsonc").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

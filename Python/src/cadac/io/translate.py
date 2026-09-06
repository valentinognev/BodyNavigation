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
    idx = line.find("//")
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
    name, op, raw_value = parts[1], parts[2], parts[3]
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
    vehicle["params"] = params
    vehicle["events"] = events
    return vehicle, i


def _parse_scenario_asc(src: Path) -> dict:
    lines = _content_lines(src)
    i = 0
    n = len(lines)
    title = ""
    options: dict[str, bool] = {}
    modules: list[dict] = []
    timing: dict[str, float] = {}
    vehicles: list[dict] = []
    end_time = 0.0
    while i < n:
        parts = lines[i].split()
        key = parts[0]
        if key == "TITLE":
            title = lines[i].split(None, 1)[1].strip() if len(parts) > 1 else ""
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
                phases = [
                    phase.strip()
                    for phase in "".join(mparts[1:]).split(",")
                    if phase.strip()
                ]
                modules.append({"name": mparts[0], "phases": phases})
                i += 1
            i += 1
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
    return {
        "title": title,
        "options": options,
        "modules": modules,
        "timing": timing,
        "end_time": end_time,
        "vehicles": vehicles,
    }


def translate_scenario_asc(src, dst_dir, family=None) -> None:
    src = Path(src)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    payload = _parse_scenario_asc(src)
    if family is not None:
        payload["family"] = family
    dst_dir.joinpath(f"{src.stem}.jsonc").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

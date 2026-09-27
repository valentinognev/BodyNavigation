import json
import re
from pathlib import Path

from cadac_cpp.extract_cpp import (
    MODE_FLAGS,
    PROGRAM_DIRS,
    _CASE,
    _MODE,
    _SWITCH,
    extract_def_modules,
    extract_modes,
    extract_vehicle_types,
)
from cadac_cpp.field_help import build_glossary
from cadac_cpp.inventory import _program_files, _read_text, _vehicle_text

FAMILIES = (
    "aim5",
    "cruise5",
    "magsix",
    "rocket6",
    "sam6",
    "sraam6",
    "agm6",
)

_OPTION = re.compile(r"=\s*(-?\d+)\s*:\s*([^;=]+)")
_ND = re.compile(r"\s*-\s*ND\s*$")
_FLAG_SET = set(MODE_FLAGS)
_NAMED = re.compile(
    r"^(?P<flag>[A-Za-z_]\w*)\s*=\s*(?P<value>-?\d+)\s*:?\s*(?P<label>.*)$"
)
_CONT_EQ = re.compile(r"^=\s*(?P<value>-?\d+)\s*:?\s*(?P<label>.+)$")
_CONT_NUM = re.compile(r"^\*?\s*(?P<value>-?\d+)\s*:\s*(?P<label>.+)$")
_CONT_GE = re.compile(r"^>=\s*(?P<value>-?\d+)\s+(?P<label>.+)$")
_BARE = re.compile(r"^(?P<flag>[A-Za-z_]\w*)\s*:?\s*$")
_PROSE = re.compile(
    r"^(?P<label>.+?):\s*(?P<flag>[A-Za-z_]\w*)\s*=\s*(?P<value>-?\d+)\s*$"
)
_HEADER = re.compile(
    r"^(FILE|Definition|Member function|Module-variable|Contains |parameter input|return output)\b",
    re.IGNORECASE,
)


def enumerated_options(sentence: str) -> list[tuple[int, str]] | None:
    if "increment" in sentence.lower():
        return None
    found: list[tuple[int, str]] = []
    seen: set[int] = set()
    for match in _OPTION.finditer(sentence):
        value = int(match.group(1))
        if value in seen:
            continue
        label = _ND.sub("", match.group(2)).strip().rstrip(",").strip()
        if label == "":
            continue
        seen.add(value)
        found.append((value, label))
    if len(found) < 2:
        return None
    found.sort(key=lambda pair: pair[0])
    return found


def _clean_label(raw: str) -> str | None:
    text = re.sub(r"\s+", " ", raw).strip(" .;")
    if text.startswith("* "):
        text = text[2:].strip()
    if not re.search(r"[A-Za-z]", text):
        return None
    if re.match(r"\d{6}\b", text) or "Created by" in text or text.startswith("//"):
        return None
    if len(text) > 160:
        return None
    return text


def _branch_label(raw: str) -> str | None:
    text = _clean_label(raw)
    if text is None or len(text) > 140 or _HEADER.match(text):
        return None
    if any(token in text for token in ("==", "{", "- start", "- end")):
        return None
    return text


def _resolve_flag(name: str) -> str | None:
    if name in _FLAG_SET:
        return name
    close = [flag for flag in MODE_FLAGS if _one_edit(name, flag)]
    if len(close) == 1:
        return close[0]
    return None


def _one_edit(left: str, right: str) -> bool:
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        diffs = [index for index, (a, b) in enumerate(zip(left, right)) if a != b]
        if len(diffs) == 1:
            return True
        return (
            len(diffs) == 2
            and diffs[1] == diffs[0] + 1
            and left[diffs[0]] == right[diffs[1]]
            and left[diffs[1]] == right[diffs[0]]
        )
    if len(left) > len(right):
        left, right = right, left
    index = 0
    skipped = False
    while index < len(left):
        if left[index] != right[index + skipped]:
            if skipped:
                return False
            skipped = True
            continue
        index += 1
    return True


def _pick_branch(lines: list[str]) -> str | None:
    for line in lines:
        folded = line.lower()
        if folded.startswith(("however", "but ", "note:", "note ", "where ")):
            continue
        return line
    return lines[-1] if lines else None


def _remember(store: dict[str, dict[int, str]], flag: str, value: int, label: str) -> None:
    cleaned = _clean_label(label)
    if cleaned is None:
        return
    store.setdefault(flag, {}).setdefault(value, cleaned)


def _store_branch(store: dict[str, dict[int, str]], flag: str, value: int, label: str) -> None:
    store.setdefault(flag, {}).setdefault(value, label)


def comment_mode_labels(text: str) -> dict[str, dict[int, str]]:
    table: dict[str, dict[int, str]] = {}
    branch: dict[str, dict[int, str]] = {}
    current: str | None = None
    pending_lines: list[str] = []
    last_codes: list[tuple[str, int]] = []
    switch_flag: str | None = None
    awaiting_brace = False
    depth = 0
    switch_depth: int | None = None

    def take_flag(name: str) -> str | None:
        return _resolve_flag(name)

    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        if stripped.startswith("//"):
            body = stripped[2:].strip()
            named = _NAMED.match(body)
            if named:
                flag = take_flag(named.group("flag"))
                if flag is not None:
                    current = flag
                    _remember(table, flag, int(named.group("value")), named.group("label"))
                pending_lines = []
                last_codes = []
                continue
            cont = _CONT_EQ.match(body) or _CONT_NUM.match(body) or _CONT_GE.match(body)
            if cont:
                if current is not None:
                    _remember(table, current, int(cont.group("value")), cont.group("label"))
                pending_lines = []
                last_codes = []
                continue
            bare = _BARE.match(body)
            if bare:
                flag = take_flag(bare.group("flag"))
                if flag is not None:
                    current = flag
                    pending_lines = []
                    last_codes = []
                    continue
            prose_body = body[2:].strip() if body.startswith("* ") else body
            prose = _PROSE.match(prose_body)
            if prose:
                flag = take_flag(prose.group("flag"))
                if flag is not None:
                    current = flag
                    _remember(table, flag, int(prose.group("value")), prose.group("label"))
                    pending_lines = []
                    last_codes = []
                    continue
            label = _branch_label(body)
            if label and last_codes and not label.lower().startswith(("however", "but ", "note:", "note ", "where ")):
                for flag, value in last_codes:
                    _store_branch(branch, flag, value, label)
                last_codes = []
                pending_lines = []
                continue
            if label:
                pending_lines.append(label)
            else:
                pending_lines = []
            last_codes = []
            continue
        code, _, trailing = stripped.partition("//")
        if switch_flag is None:
            switch_match = _SWITCH.search(code)
            if switch_match:
                switch_flag = switch_match.group(1)
                awaiting_brace = True
        matches = list(_MODE.finditer(code))
        label = _pick_branch(pending_lines) or _branch_label(trailing)
        pending_lines = []
        if len(matches) == 1 and "||" not in code and label:
            _store_branch(branch, matches[0].group(1), int(matches[0].group(2)), label)
            last_codes = []
        elif matches:
            last_codes = [(match.group(1), int(match.group(2))) for match in matches]
        else:
            last_codes = []
        if switch_flag is not None:
            case_match = _CASE.search(code)
            if case_match and label:
                _store_branch(branch, switch_flag, int(case_match.group(1)), label)
        depth += code.count("{") - code.count("}")
        if awaiting_brace and "{" in code:
            switch_depth = depth
            awaiting_brace = False
        if switch_depth is not None and depth < switch_depth:
            switch_flag = None
            switch_depth = None
            awaiting_brace = False
    merged = {flag: dict(labels) for flag, labels in table.items()}
    for flag, labels in branch.items():
        bucket = merged.setdefault(flag, {})
        for value, label in labels.items():
            bucket.setdefault(value, label)
    return merged


def merge_param_choices(
    codes: list[int] | None,
    labels: list[tuple[int, str]] | None,
    comments: dict[int, str] | None = None,
) -> list[tuple[int, str]] | None:
    by_value: dict[int, str] = {}
    if labels:
        for value, label in labels:
            by_value[value] = label
    known = set(by_value)
    if codes:
        known.update(codes)
    if comments:
        for value, label in comments.items():
            if value in known:
                by_value.setdefault(value, label)
    if codes:
        for value in codes:
            by_value.setdefault(value, str(value))
    if len(by_value) < 2:
        return None
    return [(value, by_value[value]) for value in sorted(by_value)]


def build_choices(root: Path) -> dict:
    glossary = build_glossary(root)
    types: dict[str, list[str]] = {}
    modules: dict[str, list[str]] = {}
    params: dict[str, dict[str, list[tuple[int, str]]]] = {}
    for program, rel in PROGRAM_DIRS.items():
        key = program.lower()
        files = _program_files(root / rel)
        vehicle_text, _path = _vehicle_text(files)
        types[key] = extract_vehicle_types(vehicle_text)
        names: list[str] = []
        mode_values: dict[str, list[int]] = {}
        comments: dict[str, dict[int, str]] = {}
        for path in files:
            text = _read_text(path)
            for name in extract_def_modules(text):
                if name not in names:
                    names.append(name)
            for flag, value in extract_modes(text):
                bucket = mode_values.setdefault(flag, [])
                if value not in bucket:
                    bucket.append(value)
            for flag, options in comment_mode_labels(text).items():
                stored = comments.setdefault(flag, {})
                for value, label in options.items():
                    stored.setdefault(value, label)
        modules[key] = names
        sentences = glossary.get(key, {})
        table: dict[str, list[tuple[int, str]]] = {}
        for name in sorted(set(mode_values) | set(sentences)):
            sentence = sentences.get(name)
            labels = enumerated_options(sentence) if sentence else None
            merged = merge_param_choices(mode_values.get(name), labels, comments.get(name))
            if merged is not None:
                table[name] = merged
        params[key] = table
    return {"types": types, "modules": modules, "params": params}


def render_field_choices_ts(choices: dict) -> str:
    lines = [
        "// Generated by cadac_cpp.field_choices from the twelve CADAC programs.",
        "export const FAMILIES = [",
    ]
    for name in FAMILIES:
        lines.append(f"  {json.dumps(name)},")
    lines.append("] as const;")
    lines.append("")
    lines.append('export const PHASES = ["def", "init", "exec", "term"] as const;')
    lines.append("")
    lines.append("export const vehicleTypes: Record<string, string[]> = {")
    for program in sorted(choices["types"]):
        values = ", ".join(json.dumps(value) for value in choices["types"][program])
        lines.append(f"  {program}: [{values}],")
    lines.append("};")
    lines.append("")
    lines.append("export const moduleNames: Record<string, string[]> = {")
    for program in sorted(choices["modules"]):
        values = ", ".join(json.dumps(value) for value in choices["modules"][program])
        lines.append(f"  {program}: [{values}],")
    lines.append("};")
    lines.append("")
    lines.append("export type ParamChoice = { value: number; label: string };")
    lines.append("")
    lines.append("export const paramChoices: Record<string, Record<string, ParamChoice[]>> = {")
    for program in sorted(choices["params"]):
        lines.append(f"  {program}: {{")
        for name, options in sorted(choices["params"][program].items()):
            parts = ", ".join(
                f"{{ value: {value}, label: {json.dumps(label, ensure_ascii=False)} }}"
                for value, label in options
            )
            lines.append(f"    {_ts_key(name)}: [{parts}],")
        lines.append("  },")
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def _ts_key(name: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        return name
    return json.dumps(name, ensure_ascii=False)

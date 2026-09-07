import re

PROGRAM_DIRS = {
    "HYPER3": "CADAC_Simulations/HYPER3_250114/HYPER3",
    "FALCON5": "CADAC_Simulations/FALCON5_250116/FALCON5",
    "FALCON6": "CADAC_Simulations/FALCON6_250201/FALCON6",
    "HYPER5": "CADAC_Simulations/HYPER5_250113/HYPER5",
    "HYPER6": "CADAC_Simulations/HYPER6_250125/HYPER6",
    "AIM5": "CADAC_Simulations/AIM5_250114/AIM5",
    "CRUISE5": "CADAC_Simulations/CRUISE5_250115/CRUISE5",
    "MAGSIX": "CADAC_Simulations/MAGSIX_231111/MAGSIX",
    "ROCKET6": "CADAC_Simulations/ROCKET6_250122/ROCKET6",
    "SAM6": "CADAC_Simulations/SAM6_250217/SAM6",
    "SRAAM6": "CADAC_Simulations/SRAAM6_250130/SRAAM6",
    "AGM6": "CADAC_Simulations/AGM6_250217/AGM6",
}

_STRCMP_TYPE = re.compile(
    r'^[ \t]*else[ \t]+if[ \t]*\([ \t]*!?strcmp[ \t]*\([ \t]*temp[ \t]*,[ \t]*"([^"]+)"[ \t]*\)'
    r'|^[ \t]*if[ \t]*\([ \t]*!?strcmp[ \t]*\([ \t]*temp[ \t]*,[ \t]*"([^"]+)"[ \t]*\)',
    re.MULTILINE,
)


def extract_vehicle_types(text: str) -> list[str]:
    names: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//") or stripped.startswith("/*"):
            continue
        match = _STRCMP_TYPE.search(line)
        if match:
            name = match.group(1) or match.group(2)
            if name not in names:
                names.append(name)
    return names


MODE_FLAGS = (
    "mguid", "mguidance", "mguide", "maut", "mauty", "mcontrol", "mprop",
    "mseek", "mseeker", "mins", "maero", "mact", "mtvc", "mair", "matmo",
    "mturb", "mwind", "mnav", "mterm", "mtrack", "mtarget", "skr_dyn",
    "skr_type", "minit", "mroll", "mrcs_force", "mrcs_moment", "mgps",
    "mstar", "mtargeting", "tgt_option", "acft_option", "guid_mid", "guid_term",
)

_DEF = re.compile(r"void\s+\w+::def_([A-Za-z0-9_]+)\s*\(")
_MODE = re.compile(
    r"\b(" + "|".join(MODE_FLAGS) + r")\s*==\s*(-?\d+)"
)
_SWITCH = re.compile(
    r"\bswitch\s*\(\s*(" + "|".join(MODE_FLAGS) + r")\s*\)"
)
_CASE = re.compile(r"\bcase\s+(-?\d+)\s*:")


def extract_def_modules(text: str) -> list[str]:
    names: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        match = _DEF.search(line)
        if match:
            name = match.group(1)
            if name not in names:
                names.append(name)
    return names


def extract_modes(text: str) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    live_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        live_lines.append(line)
        for match in _MODE.finditer(line):
            pair = (match.group(1), int(match.group(2)))
            if pair not in found:
                found.append(pair)
    body = "\n".join(live_lines)
    flag: str | None = None
    awaiting_brace = False
    depth = 0
    body_depth: int | None = None
    i = 0
    n = len(body)
    while i < n:
        switch_match = _SWITCH.match(body, i)
        if switch_match:
            flag = switch_match.group(1)
            awaiting_brace = True
            body_depth = None
            i = switch_match.end()
            continue
        if flag is not None and not awaiting_brace:
            case_match = _CASE.match(body, i)
            if case_match:
                pair = (flag, int(case_match.group(1)))
                if pair not in found:
                    found.append(pair)
                i = case_match.end()
                continue
        ch = body[i]
        if ch == "{":
            depth += 1
            if awaiting_brace:
                body_depth = depth
                awaiting_brace = False
        elif ch == "}":
            if body_depth is not None and depth == body_depth:
                flag = None
                body_depth = None
                awaiting_brace = False
            depth -= 1
        i += 1
    return found

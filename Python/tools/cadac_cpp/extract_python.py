import re

from cadac_cpp.extract_cpp import MODE_FLAGS

_TYPE_KEY = re.compile(r'"([A-Z][A-Z0-9]+)"\s*:')
_FAMILY = re.compile(r'\("([a-z0-9]+)",\s*"([A-Z][A-Z0-9]+)"\)')
_REGISTER = re.compile(
    r'register_family_type\(\s*"([a-z0-9]+)",\s*"([A-Z][A-Z0-9]+)"'
)
_EQ = re.compile(r"\b(" + "|".join(MODE_FLAGS) + r")\s*==\s*(-?\d+)")
_NE = re.compile(r"\b(" + "|".join(MODE_FLAGS) + r")\s*!=\s*(-?\d+)")
_IN_TUPLE = re.compile(
    r"\b(" + "|".join(MODE_FLAGS) + r")\s+not\s+in\s+\(([^)]*)\)"
    r"|\b(" + "|".join(MODE_FLAGS) + r")\s+in\s+\(([^)]*)\)"
    r"|\b(" + "|".join(MODE_FLAGS) + r")\s*!=\s*(-?\d+)"
)
_UNKNOWN = re.compile(r'unknown (' + "|".join(MODE_FLAGS) + r")")


def extract_global_types(text: str) -> list[str]:
    if "_VEHICLE_TYPES" not in text:
        return []
    start = text.index("_VEHICLE_TYPES")
    block = text[start : text.index("}", start) + 1]
    return _TYPE_KEY.findall(block)


def extract_family_keys(text: str) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for pattern in (_FAMILY, _REGISTER):
        for match in pattern.finditer(text):
            pair = (match.group(1), match.group(2))
            if pair not in keys:
                keys.append(pair)
    return keys


def extract_python_modes(text: str) -> tuple[list[tuple[str, int]], list[str]]:
    implemented: list[tuple[str, int]] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        for matcher in (_EQ, _NE):
            for match in matcher.finditer(line):
                pair = (match.group(1), int(match.group(2)))
                if pair not in implemented:
                    implemented.append(pair)
        for match in _IN_TUPLE.finditer(line):
            if match.group(5):
                pair = (match.group(5), int(match.group(6)))
                if pair not in implemented:
                    implemented.append(pair)
                continue
            flag = match.group(1) or match.group(3)
            inner = match.group(2) or match.group(4)
            for part in inner.split(","):
                part = part.strip().rstrip(",")
                if not part:
                    continue
                implemented.append((flag, int(part)))
    unique: list[tuple[str, int]] = []
    for pair in implemented:
        if pair not in unique:
            unique.append(pair)
    implemented = unique
    stubs = []
    for match in _UNKNOWN.finditer(text):
        if match.group(1) not in stubs:
            stubs.append(match.group(1))
    return implemented, stubs

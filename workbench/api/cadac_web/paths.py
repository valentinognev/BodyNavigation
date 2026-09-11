import os
import re
from pathlib import Path

from cadac.io.catalog import repo_root

_PROGRAM_RE = re.compile(r"^[a-z0-9_]+$")
_STEM_RE = re.compile(r"^[A-Za-z0-9._ -]+$")


class CasePathError(ValueError):
    """Invalid program/stem (traversal or charset)."""


def CASES_ROOT() -> Path:
    override = os.environ.get("CADAC_CASES")
    if override:
        return Path(override)
    return repo_root() / "Python" / "cases"


def resolve_case(program: str, stem: str) -> Path:
    if ".." in program or ".." in stem or "/" in program or "/" in stem or "\\" in program or "\\" in stem:
        raise CasePathError("path traversal")
    if not _PROGRAM_RE.fullmatch(program) or not _STEM_RE.fullmatch(stem):
        raise CasePathError("invalid program or stem")
    path = CASES_ROOT() / program / f"{stem}.jsonc"
    root = CASES_ROOT().resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise CasePathError("path traversal")
    return path

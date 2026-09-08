import re
from pathlib import Path

from cadac_quality.metrics import load_metrics, scan_cadac

# Iteration over a names list (caller-supplied or store.names() as the value).
_FOR_IN_NAMES = re.compile(r"\bfor\s+\w+\s+in\s+names\b")

ROOT = Path(__file__).resolve().parents[2] / "src" / "cadac"
BASELINE = Path(__file__).resolve().parents[2] / "tools" / "cadac_quality" / "baseline.json"


def test_scan_has_required_keys():
    data = scan_cadac(ROOT)
    for key in (
        "files",
        "loc",
        "code",
        "store_get",
        "store_set",
        "store_names",
        "look_up",
        "np_zeros",
        "np_array",
        "typed_defs",
        "untyped_defs",
        "empty_terminate",
        "empty_initialize",
        "skew_defs",
        "cadac_sign_defs",
    ):
        assert key in data
        assert isinstance(data[key], int)
        assert data[key] >= 0


def test_scan_matches_checked_in_baseline():
    data = scan_cadac(ROOT)
    base = load_metrics(BASELINE)
    # loc/code/untyped_defs drift with later quality tasks; store_names must drop
    # after membership moves to `in store`. Do not rewrite baseline.json.
    for key in base:
        if key in ("loc", "code", "store_names", "untyped_defs"):
            continue
        assert data[key] == base[key], key
    assert data["store_names"] < base["store_names"]


def test_src_membership_does_not_use_names_list():
    hits = []
    for path in ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if "in store.names()" in line:
                hits.append(f"{path}:{i}:{stripped}")
            if "in names" in line and "store.names()" in text:
                if _FOR_IN_NAMES.search(line):
                    continue
                hits.append(f"{path}:{i}:{stripped}")
    assert hits == [], "use `name in store`:\n" + "\n".join(hits)

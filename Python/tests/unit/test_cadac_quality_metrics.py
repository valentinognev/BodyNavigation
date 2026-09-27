import ast
import re
from pathlib import Path

from cadac_quality.metrics import _is_typed, load_metrics, scan_cadac

# Iteration over a names list (caller-supplied or store.names() as the value).
_FOR_IN_NAMES = re.compile(r"\bfor\s+\w+\s+in\s+names\b")

ROOT = Path(__file__).resolve().parents[2] / "src" / "cadac"
BASELINE = Path(__file__).resolve().parents[2] / "tools" / "cadac_quality" / "baseline.json"


def _untyped_named_defs(root: Path, wanted: dict[str, tuple[str, ...]]) -> list[str]:
    missing = []
    for rel, names in wanted.items():
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        found = {name: False for name in names}
        typed = {name: False for name in names}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in names:
                found[node.name] = True
                typed[node.name] = _is_typed(node)
        for name in names:
            if not found[name]:
                missing.append(f"{rel}:{name}:missing")
            elif not typed[name]:
                missing.append(f"{rel}:{name}:untyped")
    return missing


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
    # loc/code/untyped_defs/np_array/np_zeros drift with later quality tasks;
    # store_names and skew/sign defs drop after shared helpers.
    # typed_defs rise after kernel/math/env/eom annotations (Task 12).
    # files rose when catalog.py added a module (0.170). Do not rewrite baseline.json.
    # store_get/store_set rose with the parity ports (0.175). Do not rewrite baseline.json.
    for key in base:
        if key in (
            "files",
            "loc",
            "code",
            "store_names",
            "store_get",
            "store_set",
            "untyped_defs",
            "typed_defs",
            "skew_defs",
            "cadac_sign_defs",
            "np_array",
            "np_zeros",
            "empty_terminate",
            "empty_initialize",
        ):
            continue
        assert data[key] == base[key], key
    assert data["files"] >= base["files"]
    assert data["store_get"] >= base["store_get"]
    assert data["store_set"] >= base["store_set"]
    assert data["store_names"] < base["store_names"]
    assert data["skew_defs"] < base["skew_defs"]
    assert data["cadac_sign_defs"] < base["cadac_sign_defs"]


def test_typed_defs_exceed_baseline():
    root = Path(__file__).resolve().parents[2] / "src" / "cadac"
    data = scan_cadac(root)
    base = load_metrics(BASELINE)
    assert data["typed_defs"] > base["typed_defs"]
    missing = _untyped_named_defs(
        root,
        {
            "kernel/integrate.py": ("integrate",),
            "kernel/executive.py": ("run_loop",),
            "kernel/state.py": (
                "define",
                "get",
                "get_optional",
                "set",
                "names",
                "field",
                "__contains__",
            ),
            "math/frames.py": ("cadac_matmul",),
            "tables/lookup.py": ("look_up",),
        },
    )
    assert missing == [], "public APIs must be annotated:\n" + "\n".join(missing)


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


def test_overhaul_improved_survey_counters():
    root = Path(__file__).resolve().parents[2] / "src" / "cadac"
    now = scan_cadac(root)
    base = load_metrics(BASELINE)
    assert now["skew_defs"] == 1
    assert now["cadac_sign_defs"] == 1
    assert now["typed_defs"] > base["typed_defs"]
    assert now["store_names"] < base["store_names"]


def test_only_one_skew_and_sign_definition():
    skews, signs = [], []
    for path in ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"^def (_)?skew\(", text, re.M):
            skews.append(str(path.relative_to(ROOT)))
        if re.search(r"^def (_)?cadac_sign\(", text, re.M):
            signs.append(str(path.relative_to(ROOT)))
    assert skews == ["math/frames.py"]
    assert signs == ["math/frames.py"]

from pathlib import Path

from cadac_quality.metrics import load_metrics, scan_cadac

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
    for key in base:
        assert data[key] == base[key], key

"""SRAAM6 INLAR3WM SHAZAM (MTERM=1) e2e — python-golden."""

from pathlib import Path

import numpy as np
import pytest

from cadac import run_scenario
from cadac.io.scenario import load_scenario

CASE = Path(__file__).resolve().parents[2] / "cases" / "sraam6" / "inlar3wm.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "sraam6" / "inlar3wm.plot.csv"
RTOL = 1e-5


def _csv_atol(golden):
    return max(1e-6, 5e-6 * abs(golden))


def load_cadac_plot_csv(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    columns = [col for col in lines[2].split(",") if col]
    rows = []
    for line in lines[3:]:
        if not line.strip():
            continue
        values = [float(item) for item in line.split(",") if item != ""]
        row = dict(zip(columns, values))
        if row["time"] == -1:
            continue
        rows.append(row)
    return columns, rows


def require_golden(path: Path):
    if not path.is_file():
        pytest.skip("SRAAM6 inlar3wm python-golden CSV is absent")


def test_sraam6_inlar3wm_load_mterm_1():
    cfg = load_scenario(CASE)
    assert cfg.family == "sraam6" or cfg.vehicles[0].family == "sraam6"
    assert cfg.vehicles[0].type == "MISSILE6"
    assert cfg.vehicles[0].params["mterm"] == 1


def test_sraam6_inlar3wm_run_writes_yss_at_intercept():
    result = run_scenario(CASE)
    assert len(result.plot_rows) >= 2
    hit = [row for row in result.plot_rows if row.get("lconv") == 2]
    assert hit, "expected intercept lconv=2 under mterm=1"
    assert np.isfinite(hit[0]["yss"])
    assert np.isfinite(hit[0]["zss"])
    assert np.isfinite(hit[0]["dyrb"])
    # SHAZAM outputs non-trivial at CPA (planted geometry not zero miss)
    assert abs(hit[0]["yss"]) + abs(hit[0]["zss"]) + abs(hit[0]["dyrb"]) > 0.0


def test_sraam6_inlar3wm_shared_columns_match_golden():
    require_golden(GOLDEN)
    result = run_scenario(CASE)
    columns, golden_rows = load_cadac_plot_csv(GOLDEN)
    python_keys = result.plot_rows[0].keys()
    shared = [column for column in columns if column in python_keys]
    assert "yss" in shared
    assert "zss" in shared
    assert "dyrb" in shared

    def _row_at(rows, time):
        for row in rows:
            if abs(row["time"] - time) < 1e-9:
                return row
        raise KeyError(time)

    for golden in golden_rows:
        python = _row_at(result.plot_rows, golden["time"])
        for column in shared:
            want = golden[column]
            got = python[column]
            np.testing.assert_allclose(
                got,
                want,
                rtol=RTOL,
                atol=_csv_atol(want),
                err_msg=f"{column} at t={golden['time']}",
            )

"""CRUISE5 input_ftn_mturn0 e2e — MTURN=0 sideslip hold (python-golden)."""

from pathlib import Path

import numpy as np
import pytest

from cadac import run_scenario
from cadac.io.scenario import load_scenario

CASE = Path(__file__).resolve().parents[2] / "cases" / "cruise5" / "input_ftn_mturn0.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "cruise5" / "mturn0.plot.csv"
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
        pytest.skip("CRUISE5 mturn0 python-golden CSV is absent")


def test_cruise5_mturn0_load_scenario():
    cfg = load_scenario(CASE)
    assert cfg.family == "cruise5"
    uav = cfg.vehicles[0]
    assert uav.type == "CRUISE3"
    assert uav.params["mturn"] == 0
    assert uav.params["maut"] == 11
    assert uav.params["betac"] == 0.05
    assert cfg.end_time == 0.2


def test_cruise5_mturn0_run_holds_betax():
    """python-golden lock: short run holds BETAC as betax under MTURN=0 / MAUT=11."""
    result = run_scenario(CASE)
    assert len(result.plot_rows) >= 2
    want_betax = 0.05 * (180.0 / np.pi)
    for row in result.plot_rows:
        assert np.isfinite(row["betax"])
        assert np.isfinite(row["alphax"])
        assert np.isfinite(row["phimvx"])
        np.testing.assert_allclose(row["betax"], want_betax, rtol=1e-9, atol=1e-9)
        assert row["phimvx"] == 0.0


def test_cruise5_mturn0_shared_columns_match_golden():
    require_golden(GOLDEN)
    result = run_scenario(CASE)
    columns, golden_rows = load_cadac_plot_csv(GOLDEN)
    python_keys = result.plot_rows[0].keys()
    shared = [column for column in columns if column in python_keys]

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

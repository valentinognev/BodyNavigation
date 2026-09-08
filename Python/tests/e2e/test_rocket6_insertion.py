"""ROCKET6 insertion e2e vs optional C++ plot golden.

The golden at goldens/rocket6/plot.csv is harvested from
`input_insertion.asc` with MONTE forced to `MONTE 0 1234`. That ASC has
`mair 0` (US76, no wind, no Dryden). JSONC `iseed` is 1234.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from cadac import run_scenario
from cadac.io.jsonc import load as load_jsonc
from cadac.io.plot import write_plot_csv

CASE = Path(__file__).resolve().parents[2] / "cases" / "rocket6" / "input.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "rocket6" / "plot.csv"
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


def _row_at(rows, time):
    for row in rows:
        if abs(row["time"] - time) < 1e-9:
            return row
    raise KeyError(time)


def require_golden(path: Path):
    if not path.is_file():
        pytest.skip("ROCKET6 golden plot.csv is absent")


def _compare_alt_t0(golden_path: Path, plot_rows):
    _, golden_rows = load_cadac_plot_csv(golden_path)
    got = _row_at(plot_rows, 0.0)["alt"]
    want = _row_at(golden_rows, 0.0)["alt"]
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=_csv_atol(want))


def _compare_alt_vmach(golden_path: Path, plot_rows):
    columns, golden_rows = load_cadac_plot_csv(golden_path)
    python_keys = plot_rows[0].keys()
    shared = [column for column in columns if column in python_keys]
    assert "alt" in shared
    assert "vmach" in shared
    for golden in golden_rows:
        python = _row_at(plot_rows, golden["time"])
        for column in ("alt", "vmach"):
            want = golden[column]
            got = python[column]
            np.testing.assert_allclose(
                got,
                want,
                rtol=RTOL,
                atol=_csv_atol(want),
                err_msg=f"{column} at t={golden['time']}",
            )


def test_require_golden_skips_when_missing(tmp_path):
    with pytest.raises(pytest.skip.Exception, match="ROCKET6 golden plot.csv is absent"):
        require_golden(tmp_path / "plot.csv")


def test_require_golden_continues_when_present(tmp_path):
    path = tmp_path / "plot.csv"
    path.write_text("x\n", encoding="utf-8", newline="\n")
    require_golden(path)


def test_load_cadac_plot_csv_skips_sentinel_time(tmp_path):
    path = tmp_path / "plot.csv"
    write_plot_csv(
        path,
        "rocket6",
        ["time", "alt", "vmach"],
        [[0.0, 100.0, 0.5], [-1.0, 0.0, 0.0], [0.2, 101.0, 0.51]],
    )
    columns, rows = load_cadac_plot_csv(path)
    assert columns == ["time", "alt", "vmach"]
    assert [row["time"] for row in rows] == [0.0, 0.2]


def test_alt_t0_matches_within_csv_tolerances(tmp_path):
    path = tmp_path / "plot.csv"
    write_plot_csv(path, "rocket6", ["time", "alt"], [[0.0, 100.0]])
    _compare_alt_t0(path, [{"time": 0.0, "alt": 100.0}])


def test_alt_t0_rejects_outside_csv_tolerances(tmp_path):
    path = tmp_path / "plot.csv"
    write_plot_csv(path, "rocket6", ["time", "alt"], [[0.0, 100.0]])
    with pytest.raises(AssertionError):
        _compare_alt_t0(path, [{"time": 0.0, "alt": 0.0}])


def test_shared_columns_ignore_names_only_on_one_side(tmp_path):
    path = tmp_path / "plot.csv"
    write_plot_csv(
        path,
        "rocket6",
        ["time", "alt", "vmach", "lonx"],
        [[0.0, 100.0, 0.5, -120.6]],
    )
    _compare_alt_vmach(
        path,
        [{"time": 0.0, "alt": 100.0, "vmach": 0.5, "SBEL1": 0.0, "hbe": 0.0}],
    )


def test_shared_columns_mismatch_fails(tmp_path):
    path = tmp_path / "plot.csv"
    write_plot_csv(path, "rocket6", ["time", "alt", "vmach"], [[0.0, 100.0, 0.5]])
    with pytest.raises(AssertionError):
        _compare_alt_vmach(path, [{"time": 0.0, "alt": 100.0, "vmach": 0.0}])


def test_shared_columns_require_alt_and_vmach(tmp_path):
    path = tmp_path / "plot.csv"
    write_plot_csv(path, "rocket6", ["time", "alt", "mach"], [[0.0, 100.0, 0.5]])
    with pytest.raises(AssertionError):
        _compare_alt_vmach(
            path,
            [{"time": 0.0, "alt": 100.0, "mach": 0.5}],
        )


def test_alt_matches_golden_at_t0():
    require_golden(GOLDEN)
    result = run_scenario(CASE)
    _compare_alt_t0(GOLDEN, result.plot_rows)


def _compare_column_at(golden_path: Path, plot_rows, column, time):
    _, golden_rows = load_cadac_plot_csv(golden_path)
    got = _row_at(plot_rows, time)[column]
    want = _row_at(golden_rows, time)[column]
    np.testing.assert_allclose(
        got,
        want,
        rtol=RTOL,
        atol=_csv_atol(want),
        err_msg=f"{column} at t={time}",
    )


def _short_insertion_case(tmp_path: Path, end_time=0.1) -> Path:
    for name in ("aero_deck_SLV.jsonc", "weather_deck_Wallops.jsonc"):
        shutil.copy(CASE.parent / name, tmp_path / name)
    data = load_jsonc(CASE)
    data["end_time"] = end_time
    path = tmp_path / "input.jsonc"
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def test_vmach_matches_golden_at_t01(tmp_path):
    require_golden(GOLDEN)
    result = run_scenario(_short_insertion_case(tmp_path, 0.1))
    _compare_column_at(GOLDEN, result.plot_rows, "vmach", 0.1)


def test_ins_euler_matches_golden_at_t0(tmp_path):
    """Pitch-90 /EPS yaw: cad_geo84_in TDI vs kinematics TDI must not yield ~66°."""
    require_golden(GOLDEN)
    result = run_scenario(_short_insertion_case(tmp_path, 0.0))
    _compare_column_at(GOLDEN, result.plot_rows, "psibdcx", 0.0)
    _compare_column_at(GOLDEN, result.plot_rows, "phibdcx", 0.0)


def test_ins_pos_err_matches_golden_at_t0(tmp_path):
    """C++ mins=1 init_ins Cholesky; Python must not zero ESBI (golden ~16.13 m)."""
    require_golden(GOLDEN)
    result = run_scenario(_short_insertion_case(tmp_path, 0.0))
    _compare_column_at(GOLDEN, result.plot_rows, "ins_pos_err", 0.0)
    _compare_column_at(GOLDEN, result.plot_rows, "ESBI1", 0.0)
    _compare_column_at(GOLDEN, result.plot_rows, "ESBI2", 0.0)
    _compare_column_at(GOLDEN, result.plot_rows, "ESBI3", 0.0)


def test_vmach_matches_golden_at_t1297(tmp_path):
    """US76 geometric 84.852 km ceiling: golden pdynmc=0 and vmach jump at t=129.7."""
    require_golden(GOLDEN)
    result = run_scenario(_short_insertion_case(tmp_path, 129.7))
    _compare_column_at(GOLDEN, result.plot_rows, "vmach", 129.7)


def test_alt_matches_golden_at_t1412(tmp_path):
    """Post-vacuum LTG/thrust pointing: golden alt at t=141.2 (still modes 5004)."""
    require_golden(GOLDEN)
    result = run_scenario(_short_insertion_case(tmp_path, 141.2))
    _compare_column_at(GOLDEN, result.plot_rows, "alt", 141.2)


def test_alt_matches_golden_at_t1565(tmp_path):
    """Post-141.2 pitch RCS residual: golden alt at t=156.5 (still modes 5004)."""
    require_golden(GOLDEN)
    result = run_scenario(_short_insertion_case(tmp_path, 156.5))
    _compare_column_at(GOLDEN, result.plot_rows, "alt", 156.5)


def test_vmach_matches_golden_at_t1824(tmp_path):
    """LTG beco in the t=182.3–182.4 bin: golden vmach after cutoff."""
    require_golden(GOLDEN)
    result = run_scenario(_short_insertion_case(tmp_path, 182.4))
    _compare_column_at(GOLDEN, result.plot_rows, "vmach", 182.4)


def test_alt_and_vmach_match_golden_at_shared_times():
    require_golden(GOLDEN)
    result = run_scenario(CASE)
    _compare_alt_vmach(GOLDEN, result.plot_rows)

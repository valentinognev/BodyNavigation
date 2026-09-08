"""SAM6 autopilot e2e vs harvested golden plot.csv.

Skip if tests/e2e/goldens/sam6/plot.csv is absent. Harvest used nmonte=0 and
default iseed=0; C++ gauss()/uniform() in init_ins and ins() still draw.
"""

from pathlib import Path

import numpy as np
import pytest

from cadac import run_scenario
from cadac.io.plot import write_plot_csv

CASE = Path(__file__).resolve().parents[2] / "cases" / "sam6" / "input_SAM_autopilot.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "sam6" / "plot.csv"
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
        pytest.skip("SAM6 golden plot.csv is absent")


def _compare_time_t0(golden_path: Path, plot_rows):
    _, golden_rows = load_cadac_plot_csv(golden_path)
    got = _row_at(plot_rows, 0.0)["time"]
    want = _row_at(golden_rows, 0.0)["time"]
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=_csv_atol(want))


def _compare_shared_columns(golden_path: Path, plot_rows):
    columns, golden_rows = load_cadac_plot_csv(golden_path)
    python_keys = plot_rows[0].keys()
    shared = [column for column in columns if column in python_keys]
    for golden in golden_rows:
        python = _row_at(plot_rows, golden["time"])
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


def test_require_golden_skips_when_missing(tmp_path):
    with pytest.raises(pytest.skip.Exception, match="SAM6 golden plot.csv is absent"):
        require_golden(tmp_path / "plot.csv")


def test_require_golden_continues_when_present(tmp_path):
    path = tmp_path / "plot.csv"
    path.write_text("x\n", encoding="utf-8", newline="\n")
    require_golden(path)


def test_load_cadac_plot_csv_skips_sentinel_time(tmp_path):
    path = tmp_path / "plot.csv"
    write_plot_csv(
        path,
        "sam6",
        ["time", "hbe"],
        [[0.0, 0.0], [-1.0, 0.0], [0.2, 1.0]],
    )
    columns, rows = load_cadac_plot_csv(path)
    assert columns == ["time", "hbe"]
    assert [row["time"] for row in rows] == [0.0, 0.2]


def test_time_t0_matches_within_csv_tolerances(tmp_path):
    path = tmp_path / "plot.csv"
    write_plot_csv(path, "sam6", ["time"], [[0.0]])
    _compare_time_t0(path, [{"time": 0.0}])


def test_time_t0_rejects_missing_t0_row(tmp_path):
    path = tmp_path / "plot.csv"
    write_plot_csv(path, "sam6", ["time"], [[0.0]])
    with pytest.raises(KeyError):
        _compare_time_t0(path, [{"time": 1.0}])


def test_shared_columns_ignore_names_only_on_one_side(tmp_path):
    path = tmp_path / "plot.csv"
    write_plot_csv(
        path,
        "sam6",
        ["time", "hbe", "lonx"],
        [[0.0, 0.0, -80.55]],
    )
    _compare_shared_columns(
        path,
        [{"time": 0.0, "hbe": 0.0, "SBEL1": 0.0}],
    )


def test_shared_columns_mismatch_fails(tmp_path):
    path = tmp_path / "plot.csv"
    write_plot_csv(path, "sam6", ["time", "hbe"], [[0.0, 0.0]])
    with pytest.raises(AssertionError):
        _compare_shared_columns(path, [{"time": 0.0, "hbe": 100.0}])


def test_time_matches_golden_at_t0():
    require_golden(GOLDEN)
    result = run_scenario(CASE)
    _compare_time_t0(GOLDEN, result.plot_rows)


def test_thtvlcx_matches_golden_at_t0():
    require_golden(GOLDEN)
    result = run_scenario(CASE)
    _, golden_rows = load_cadac_plot_csv(GOLDEN)
    want = _row_at(golden_rows, 0.0)["thtvlcx"]
    got = _row_at(result.plot_rows, 0.0)["thtvlcx"]
    np.testing.assert_allclose(
        got,
        want,
        rtol=RTOL,
        atol=_csv_atol(want),
        err_msg="thtvlcx at t=0",
    )


def test_shared_missile_plot_columns_match_golden():
    require_golden(GOLDEN)
    result = run_scenario(CASE)
    _compare_shared_columns(GOLDEN, result.plot_rows)

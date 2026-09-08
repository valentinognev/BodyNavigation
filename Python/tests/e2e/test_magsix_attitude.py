from pathlib import Path

import numpy as np
import pytest

from cadac import run_scenario
from cadac.io.plot import write_plot_csv

CASE = Path(__file__).resolve().parents[2] / "cases" / "magsix" / "input.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "magsix" / "plot.csv"
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


def _align_key(row):
    return "sim_time" if "sim_time" in row else "time"


def _row_at(rows, key, value):
    for row in rows:
        if abs(row[key] - value) < 1e-9:
            return row
    raise KeyError((key, value))


def require_golden(path: Path):
    if not path.is_file():
        pytest.skip("MAGSIX attitude golden plot.csv is absent")


def test_require_golden_skips_when_missing(tmp_path):
    with pytest.raises(pytest.skip.Exception, match="MAGSIX attitude golden"):
        require_golden(tmp_path / "plot.csv")


def test_hbe_t0_matches_within_csv_tolerances(tmp_path):
    path = tmp_path / "plot.csv"
    write_plot_csv(path, "magsix", ["time", "sim_time", "hbe"], [[0.0, 0.0, 1000.0]])
    require_golden(path)
    _, golden_rows = load_cadac_plot_csv(path)
    key = _align_key(golden_rows[0])
    got = _row_at([{"time": 0.0, "sim_time": 0.0, "hbe": 1000.0}], key, 0.0)["hbe"]
    want = _row_at(golden_rows, key, 0.0)["hbe"]
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=_csv_atol(want))


def test_hbe_matches_golden_at_t0():
    require_golden(GOLDEN)
    result = run_scenario(CASE)
    _, golden_rows = load_cadac_plot_csv(GOLDEN)
    key = _align_key(golden_rows[0]) if "sim_time" in result.plot_rows[0] else "time"
    got = _row_at(result.plot_rows, key, 0.0)["hbe"]
    want = _row_at(golden_rows, key, 0.0)["hbe"]
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=_csv_atol(want))


def test_python_plot_includes_golden_last_int_step_sim_time():
    require_golden(GOLDEN)
    _, golden_rows = load_cadac_plot_csv(GOLDEN)
    golden_times = [row["sim_time"] for row in golden_rows]
    assert any(abs(t - 0.35) < 1e-9 for t in golden_times)
    assert any(abs(t - 0.3501) < 1e-9 for t in golden_times)
    result = run_scenario(CASE)
    python_times = [row["sim_time"] for row in result.plot_rows]
    assert any(abs(t - 0.35) < 1e-9 for t in python_times)
    assert any(abs(t - 0.3501) < 1e-9 for t in python_times)


def test_all_shared_plot_columns_match_golden_at_shared_times():
    require_golden(GOLDEN)
    result = run_scenario(CASE)
    columns, golden_rows = load_cadac_plot_csv(GOLDEN)
    python_keys = result.plot_rows[0].keys()
    shared = [column for column in columns if column in python_keys]
    assert "hbe" in shared
    assert "dvbe" in shared
    assert "sim_time" in shared
    key = "sim_time" if "sim_time" in shared else "time"
    compare = [column for column in shared if column != key]
    assert compare
    for golden in golden_rows:
        python = _row_at(result.plot_rows, key, golden[key])
        for column in compare:
            want = golden[column]
            np.testing.assert_allclose(
                python[column],
                want,
                rtol=RTOL,
                atol=_csv_atol(want),
                err_msg=f"{column} at {key}={golden[key]}",
            )

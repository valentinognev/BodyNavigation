from pathlib import Path

import numpy as np

from cadac import run_scenario

CASE = Path(__file__).resolve().parents[2] / "cases" / "hyper3" / "input_climb.jsonc"
GOLDEN = Path(__file__).resolve().parent / "goldens" / "hyper3" / "plot1.csv"
RTOL = 1e-5
PLOT_COLUMNS = [
    "time",
    "FSPV1",
    "FSPV2",
    "FSPV3",
    "pdynmc",
    "mach",
    "lonx",
    "latx",
    "alt",
    "dvbe",
    "psivgx",
    "thtvgx",
    "SBEG1",
    "SBEG2",
    "SBEG3",
    "VBEG1",
    "VBEG2",
    "VBEG3",
    "throttle",
    "mass",
    "thrust",
    "fmassr",
    "cl_ov_cd",
]


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


def test_alt_matches_golden_at_t0_and_t02():
    _, golden_rows = load_cadac_plot_csv(GOLDEN)
    result = run_scenario(CASE)
    for time in (0.0, 0.2):
        got = _row_at(result.plot_rows, time)["alt"]
        want = _row_at(golden_rows, time)["alt"]
        np.testing.assert_allclose(got, want, rtol=RTOL, atol=_csv_atol(want))


def test_climb_plot_grid_matches_golden():
    columns, golden_rows = load_cadac_plot_csv(GOLDEN)
    assert columns == PLOT_COLUMNS
    result = run_scenario(CASE)
    assert len(result.plot_rows) == len(golden_rows)
    for golden in golden_rows:
        python = _row_at(result.plot_rows, golden["time"])
        for column in PLOT_COLUMNS:
            want = golden[column]
            got = python[column]
            np.testing.assert_allclose(
                got,
                want,
                rtol=RTOL,
                atol=_csv_atol(want),
                err_msg=f"{column} at t={golden['time']}",
            )

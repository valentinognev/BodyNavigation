from pathlib import Path

from cadac.io.plot import plot_row, write_plot_csv
from cadac.kernel.state import Field, StateStore


def test_roundtrip_two_rows_time_alt(tmp_path: Path):
    path = tmp_path / "plot.csv"
    write_plot_csv(
        path,
        "HYPER3 climb",
        ["time", "alt"],
        [[0.0, 3000.0], [0.2, 3000.23]],
    )
    raw = path.read_bytes()
    assert b"\r" not in raw
    lines = raw.decode("utf-8").split("\n")
    assert lines[0] == "HYPER3 climb"
    assert lines[1].split() == ["0", "0", "2"]
    assert lines[2].endswith(",")
    columns = [c for c in lines[2].split(",") if c != ""]
    assert columns == ["time", "alt"]
    rows = []
    for line in lines[3:]:
        if line == "":
            continue
        assert line.endswith(",")
        rows.append([float(x) for x in line.split(",") if x != ""])
    assert rows == [[0.0, 3000.0], [0.2, 3000.23]]


def test_plot_row_columns_argument_reads_named_fields():
    store = StateStore()
    store.define(Field("time", 1.0, "real", "exec", "kinematics", ("plot",)))
    store.define(Field("alt", 3000.0, "real", "out", "newton", ("plot",)))
    store.define(Field("FSPV", (1.0, 2.0, 3.0), "vec", "out", "forces", ("plot",)))
    row = plot_row(store, columns=["time", "alt", "FSPV1"])
    assert row["time"] == 1.0
    assert row["alt"] == 3000.0
    assert row["FSPV1"] == 1.0
    assert "lonx" not in row


def test_plot_row_expands_vec_name123():
    store = StateStore()
    store.define(Field("SBEL", (10.0, 20.0, -3500.0), "vec", "state", "newton", ("plot",)))
    row = plot_row(store, columns=["SBEL1", "SBEL2", "SBEL3"])
    assert row["SBEL1"] == 10.0
    assert row["SBEL2"] == 20.0
    assert row["SBEL3"] == -3500.0

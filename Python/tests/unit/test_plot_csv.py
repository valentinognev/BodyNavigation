from pathlib import Path

from cadac.io.plot import write_plot_csv


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

import os
from pathlib import Path

import pytest

from cadac_cpp.harvest import (
    csv_from_plot_asc,
    find_plot_csv,
    force_csv_on,
    force_monte_off,
    harvest_all,
    harvest_row,
    restore_cadac_io,
    snapshot_cadac_io,
)
from cadac_cpp.harvest_table import HARVEST_ROWS, repo_root

_HARVEST = pytest.mark.skipif(
    os.environ.get("CADAC_HARVEST") != "1",
    reason="set CADAC_HARVEST=1 to rebuild/run CADAC harvest",
)


def load_cadac_plot_csv(path):
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


def test_force_monte_off():
    text = "MONTE 1\nnmonte 5\n"
    out = force_monte_off(text)
    assert "MONTE 0" in out or "n_monte 0" in out or "nmonte 0" in out.lower()
    assert "MONTE 1" not in out


def test_force_csv_on():
    replaced = force_csv_on("OPTIONS y_scrn y_plot n_csv\n")
    assert "y_csv" in replaced
    assert "n_csv" not in replaced
    appended = force_csv_on("OPTIONS y_scrn y_plot\n")
    assert "y_csv" in appended
    already = force_csv_on("OPTIONS y_scrn y_plot y_csv\n")
    assert already.count("y_csv") == 1


def test_csv_from_plot_asc_matches_cadac_csv(tmp_path):
    src_asc = Path(__file__).parent / "fixtures" / "cadac_plot1.asc"
    if not src_asc.is_file():
        pytest.skip("missing Python/tests/unit/fixtures/cadac_plot1.asc")
    copied = tmp_path / "plot1.asc"
    copied.write_bytes(src_asc.read_bytes())
    produced = csv_from_plot_asc(copied)
    lines = produced.read_text(encoding="utf-8", errors="replace").splitlines()
    assert lines[2].split(",")[:5] == ["time", "a", "b", "c", "d"]
    assert lines[3].split(",")[:5] == ["0", "1", "2", "3", "4"]
    assert lines[4].split(",")[:5] == ["0.2", "5", "6", "7", "8"]


def test_find_plot_prefers_plot1(tmp_path):
    (tmp_path / "plot.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "plot1.csv").write_text("y\n", encoding="utf-8")
    assert find_plot_csv(tmp_path).name == "plot1.csv"


def test_restore_cadac_io_restores_preexisting_files(tmp_path):
    (tmp_path / "doc.asc").write_text("orig-doc\n", encoding="utf-8")
    (tmp_path / "input.asc").write_text("orig-input\n", encoding="utf-8")
    (tmp_path / "plot1.csv").write_text("orig-plot\n", encoding="utf-8")
    snap = snapshot_cadac_io(tmp_path)
    (tmp_path / "doc.asc").write_text("dirty\n", encoding="utf-8")
    (tmp_path / "plot1.csv").write_text("dirty\n", encoding="utf-8")
    (tmp_path / "tabout.asc").write_text("leftover\n", encoding="utf-8")
    restore_cadac_io(tmp_path, snap)
    assert (tmp_path / "doc.asc").read_text(encoding="utf-8") == "orig-doc\n"
    assert (tmp_path / "input.asc").read_text(encoding="utf-8") == "orig-input\n"
    assert (tmp_path / "plot1.csv").read_text(encoding="utf-8") == "orig-plot\n"
    assert (tmp_path / "tabout.asc").read_text(encoding="utf-8") == "leftover\n"


@_HARVEST
@pytest.mark.integration
def test_hyper3_harvest_matches_checked_in_golden():
    import numpy as np

    row = next(r for r in HARVEST_ROWS if r.golden.endswith("hyper3/plot1.csv"))
    produced = harvest_row(row)
    golden = repo_root() / row.golden
    _, got_rows = load_cadac_plot_csv(produced)
    _, want_rows = load_cadac_plot_csv(golden)
    assert len(got_rows) == len(want_rows)
    for got, want in zip(got_rows, want_rows):
        np.testing.assert_allclose(
            got["alt"], want["alt"],
            rtol=1e-5, atol=max(1e-6, 5e-6 * abs(want["alt"])),
        )


@_HARVEST
@pytest.mark.integration
def test_harvest_all_writes_or_skips():
    results = harvest_all()
    assert set(results) == {r.golden for r in HARVEST_ROWS}
    root = repo_root()
    for golden, status in results.items():
        assert status in {"ok", "build_failed", "run_failed"}
        if status == "ok":
            assert (root / golden).is_file()

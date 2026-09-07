from pathlib import Path

from cadac_cpp.harvest_table import HARVEST_ROWS, repo_root


def test_fourteen_harvest_rows():
    assert len(HARVEST_ROWS) == 14


def test_hyper3_canary_mapping():
    row = next(r for r in HARVEST_ROWS if "hyper3" in r.jsonc)
    root = repo_root()
    assert row.asc_name == "input_climb.asc"
    assert (root / row.cpp_dir / row.asc_name).is_file()
    assert row.golden == "Python/tests/e2e/goldens/hyper3/plot1.csv"


def test_hyper6_sat_type_not_in_harvest():
    assert all("RADAR" not in r.asc_name for r in HARVEST_ROWS)


def test_agm6_and_magsix_titles():
    names = {r.asc_name for r in HARVEST_ROWS}
    assert "input_3_1 AGM6 Free Flight.asc" in names
    assert "input_2_1 AGM6 Test Case.asc" in names
    assert "input_attitudeMR1.asc" in names
    assert "input_trajectoryMR1.asc" in names
    assert "input_Demo_4_7_pro_nav.asc" in names
    assert "input_insertion.asc" in names

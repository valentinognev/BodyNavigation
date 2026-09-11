from cadac.aero_map.payload import payload_from_aid, payload_from_mdt_rows


def test_mdt_rows_grid():
    rows = [
        {"mach": 0.8, "alpha": 0.0, "cn": 0.1, "cm": -0.01, "ca": 0.3},
        {"mach": 0.8, "alpha": 4.0, "cn": 0.5, "cm": -0.02, "ca": 0.31},
        {"mach": 1.2, "alpha": 0.0, "cn": 0.12, "cm": -0.011, "ca": 0.4},
        {"mach": 1.2, "alpha": 4.0, "cn": 0.6, "cm": -0.03, "ca": 0.41},
    ]
    p = payload_from_mdt_rows(rows)
    assert p.source == "misdc"
    assert p.solver == "mdt"
    assert p.axes["mach"] == [0.8, 1.2]
    assert p.axes["alpha"] == [0.0, 4.0]
    assert p.tables["cn"][0][1] == 0.5


def test_nan_cn_omits_cn_table():
    rows = [{"mach": 0.5, "alpha": 0.0, "cn": float("nan"), "cm": 0.0, "ca": 0.2}]
    p = payload_from_mdt_rows(rows)
    assert "cn" not in p.tables
    assert "ca" in p.tables


def test_aid_lists():
    p = payload_from_aid({"alpha": [-2.0, 0.0, 4.0], "CL": [0.0, 0.2, 0.6], "CD": [0.02, 0.02, 0.04], "Cm": [0.0, -0.01, -0.03], "MACH": 0.2})
    assert p.source == "aid"
    assert p.tables["cl"][0][1] == 0.2


def test_empty_mach_slice_dropped():
    rows = [
        {"mach": 0.5, "alpha": 0.0, "cn": float("nan"), "cm": float("inf"), "ca": float("-inf")},
        {"mach": 1.2, "alpha": 0.0, "cn": 0.1, "cm": 0.0, "ca": 0.2},
    ]
    p = payload_from_mdt_rows(rows)
    assert p.axes["mach"] == [1.2]
    assert p.tables["cn"] == [[0.1]]
    assert p.tables["ca"] == [[0.2]]


def test_empty_mach_orphan_alpha_keeps_finite_tables():
    rows = [
        {"mach": 0.5, "alpha": 2.0, "cn": float("nan"), "cm": float("nan"), "ca": float("nan")},
        {"mach": 1.2, "alpha": 0.0, "cn": 0.1, "cm": 0.0, "ca": 0.2},
    ]
    p = payload_from_mdt_rows(rows)
    assert p.axes["mach"] == [1.2]
    assert p.axes["alpha"] == [0.0]
    assert p.tables["cn"] == [[0.1]]
    assert p.tables["ca"] == [[0.2]]

from cadac.aero_map import map_payload, payload_from_mdt_rows

GRID_ROWS = [
    {"mach": 0.8, "alpha": 0.0, "cn": 0.1, "cm": -0.01, "ca": 0.3},
    {"mach": 0.8, "alpha": 4.0, "cn": 0.5, "cm": -0.02, "ca": 0.31},
    {"mach": 1.2, "alpha": 0.0, "cn": 0.12, "cm": -0.011, "ca": 0.4},
    {"mach": 1.2, "alpha": 4.0, "cn": 0.6, "cm": -0.03, "ca": 0.41},
]


def _table(deck, name):
    return next(t for t in deck["tables"] if t["name"] == name)


def test_aim5_cl_from_cn():
    payload = payload_from_mdt_rows(GRID_ROWS)
    assert "cl" not in payload.tables
    assert "cd" not in payload.tables
    result = map_payload("aim5", "AIM5", payload, None)
    by = {r.name: r.status for r in result.rows}
    assert by["cl_aim_vs_alpha_mach"] == "mapped"
    assert by["cd_aim_on_vs_alpha_mach"] == "mapped"
    assert by["cd_aim_off_vs_alpha_mach"] == "mapped"
    assert result.can_confirm is True

    cl = _table(result.deck, "cl_aim_vs_alpha_mach")
    assert cl["dim"] == 2
    assert cl["x1"] == [0.0, 4.0]
    assert cl["x2"] == [0.8, 1.2]
    assert cl["values"] == [[0.1, 0.12], [0.5, 0.6]]

    cd_on = _table(result.deck, "cd_aim_on_vs_alpha_mach")
    cd_off = _table(result.deck, "cd_aim_off_vs_alpha_mach")
    assert cd_on["values"] == [[0.3, 0.4], [0.31, 0.41]]
    assert cd_off["values"] == cd_on["values"]
    assert cd_off["values"] is not cd_on["values"]


def test_aim5_prefers_cl_cd_over_cn_ca():
    rows = [
        {
            "mach": 0.8,
            "alpha": 0.0,
            "cn": 9.0,
            "cm": 0.0,
            "ca": 9.0,
            "cl": 0.2,
            "cd": 0.05,
        },
        {
            "mach": 0.8,
            "alpha": 4.0,
            "cn": 9.0,
            "cm": 0.0,
            "ca": 9.0,
            "cl": 0.7,
            "cd": 0.08,
        },
        {
            "mach": 1.2,
            "alpha": 0.0,
            "cn": 9.0,
            "cm": 0.0,
            "ca": 9.0,
            "cl": 0.25,
            "cd": 0.06,
        },
        {
            "mach": 1.2,
            "alpha": 4.0,
            "cn": 9.0,
            "cm": 0.0,
            "ca": 9.0,
            "cl": 0.8,
            "cd": 0.09,
        },
    ]
    result = map_payload("aim5", "AIM5", payload_from_mdt_rows(rows), None)
    cl = _table(result.deck, "cl_aim_vs_alpha_mach")
    assert cl["values"] == [[0.2, 0.25], [0.7, 0.8]]
    cd_on = _table(result.deck, "cd_aim_on_vs_alpha_mach")
    assert cd_on["values"] == [[0.05, 0.06], [0.08, 0.09]]

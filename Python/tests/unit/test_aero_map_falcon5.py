from cadac.aero_map import map_payload, payload_from_aid

AID = {
    "alpha": [-2.0, 0.0, 4.0],
    "CL": [0.0, 0.2, 0.6],
    "CD": [0.02, 0.02, 0.04],
    "Cm": [0.0, -0.01, -0.03],
    "MACH": 0.2,
}

_MACS = ("30MAC", "35MAC", "40MAC")


def _table(deck, name):
    return next(t for t in deck["tables"] if t["name"] == name)


def test_falcon5_copies_mac():
    result = map_payload(None, "PLANE", payload_from_aid(AID), None)
    assert result.can_confirm is True
    names = {t["name"] for t in result.deck["tables"]}
    assert "cl_40MAC_vs_mach_alphax" in names
    by = {r.name: r.status for r in result.rows}
    for mac in _MACS:
        assert by[f"cl_{mac}_vs_mach_alphax"] == "mapped"
        assert by[f"cd_{mac}_vs_mach_alphax"] == "mapped"

    cl30 = _table(result.deck, "cl_30MAC_vs_mach_alphax")
    assert cl30["dim"] == 2
    assert cl30["x1"] == [0.2]
    assert cl30["x2"] == [-2.0, 0.0, 4.0]
    assert cl30["values"] == [[0.0, 0.2, 0.6]]

    cd30 = _table(result.deck, "cd_30MAC_vs_mach_alphax")
    assert cd30["dim"] == 2
    assert cd30["x1"] == [0.2]
    assert cd30["x2"] == [-2.0, 0.0, 4.0]
    assert cd30["values"] == [[0.02, 0.02, 0.04]]

    for mac in ("35MAC", "40MAC"):
        cl = _table(result.deck, f"cl_{mac}_vs_mach_alphax")
        cd = _table(result.deck, f"cd_{mac}_vs_mach_alphax")
        assert cl["values"] == cl30["values"]
        assert cd["values"] == cd30["values"]
        assert cl["values"] is not cl30["values"]
        assert cd["values"] is not cd30["values"]

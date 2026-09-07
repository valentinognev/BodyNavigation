from cadac_cpp.schema import InventoryRow, dump_inventory, load_inventory, KINDS, STATUSES


def test_roundtrip(tmp_path):
    row = InventoryRow(
        program="HYPER6",
        kind="vehicle",
        name="RADAR0",
        cpp="HYPER6/global_functions.cpp:set_obj_type",
        python=None,
        status="missing",
        note="not registered",
    )
    path = tmp_path / "inventory.json"
    dump_inventory(path, [row])
    got = load_inventory(path)
    assert got == [row]


def test_kinds_and_statuses():
    assert KINDS == ("vehicle", "module", "mode", "kernel", "e2e", "harvest")
    assert STATUSES == ("ported", "stubbed", "missing", "deferred", "diverged")


def test_rejects_bad_status():
    try:
        InventoryRow("HYPER3", "vehicle", "CRUISE3", "x", "y", "nope", "")
    except ValueError as exc:
        assert "status" in str(exc)
    else:
        raise AssertionError("expected ValueError")

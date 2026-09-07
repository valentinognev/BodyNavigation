from cadac_cpp.inventory import apply_human_notes, status_for_mode, scan_all
from cadac_cpp.harvest_table import repo_root
from cadac_cpp.kernel_rows import kernel_rows
from cadac_cpp.schema import InventoryRow, load_inventory


def test_status_for_mode():
    assert status_for_mode(("maut", 24), {("maut", 24)}, ["maut"]) == "ported"
    assert status_for_mode(("maut", 99), {("maut", 24)}, ["maut"]) == "stubbed"
    assert status_for_mode(("mguide", 5), set(), []) == "missing"


def test_kernel_includes_deferred_monte_and_ported_lookup():
    rows = {r.name: r for r in kernel_rows()}
    assert rows["look_up"].status == "ported"
    assert rows["nmonte"].status == "deferred"
    assert rows["scrn"].status == "deferred"


def test_scan_hyper6_radar_missing():
    rows = scan_all(repo_root())
    radar = [
        r for r in rows
        if r.program == "HYPER6" and r.kind == "vehicle" and r.name == "RADAR0"
    ]
    assert len(radar) == 1
    assert radar[0].status == "missing"
    sat3 = [
        r for r in rows
        if r.program == "HYPER6" and r.kind == "vehicle" and r.name == "SAT3"
    ]
    assert sat3[0].status == "missing"


def test_inventory_has_e2e_and_harvest_kinds():
    path = repo_root() / "Python/tools/cadac_cpp/inventory.json"
    rows = load_inventory(path)
    kinds = {r.kind for r in rows}
    assert "harvest" in kinds
    assert "e2e" in kinds


def test_missing_mode_note_is_dropout():
    rows = apply_human_notes(
        [
            InventoryRow("HYPER6", "mode", "mguide=5", "HYPER6/x.cpp", None, "missing", ""),
            InventoryRow("HYPER6", "mode", "mauty=3", "HYPER6/x.cpp", None, "missing", ""),
            InventoryRow("HYPER6", "vehicle", "RADAR0", "HYPER6/g.cpp", None, "missing", ""),
            InventoryRow("HYPER6", "vehicle", "SAT3", "HYPER6/g.cpp", None, "missing", ""),
            InventoryRow("HYPER6", "mode", "mprop=9", "HYPER6/x.cpp", None, "stubbed", ""),
        ]
    )
    by = {(r.kind, r.name): r for r in rows}
    assert by[("mode", "mguide=5")].note == "drop-out"
    assert by[("mode", "mguide=5")].status == "missing"
    assert by[("mode", "mauty=3")].note == "slice limit"
    assert by[("mode", "mauty=3")].status == "missing"
    assert by[("vehicle", "RADAR0")].note == "slice limit"
    assert by[("vehicle", "RADAR0")].status == "missing"
    assert by[("vehicle", "SAT3")].status == "missing"
    assert by[("mode", "mprop=9")].note == ""


def test_cross_family_module_note_is_name_match_only():
    rows = apply_human_notes(
        [
            InventoryRow(
                "HYPER6", "module", "seeker", "HYPER6/seeker_modules.cpp",
                "cadac.vehicles.aim5.seeker", "ported", "",
            ),
            InventoryRow(
                "AIM5", "module", "seeker", "AIM5/seeker_modules.cpp",
                "cadac.vehicles.aim5.seeker", "ported", "",
            ),
            InventoryRow(
                "HYPER6", "module", "kinematics", "HYPER6/ground0_modules.cpp",
                "cadac.eom.flat0", "ported", "",
            ),
        ]
    )
    by = {(r.program, r.name): r for r in rows}
    assert by[("HYPER6", "seeker")].note == "name match only"
    assert by[("HYPER6", "seeker")].status == "ported"
    assert by[("AIM5", "seeker")].note == ""
    assert by[("HYPER6", "kinematics")].note == ""


def test_inventory_missing_modes_dropout_or_slice_limit():
    rows = load_inventory(repo_root() / "Python/tools/cadac_cpp/inventory.json")
    missing_modes = [r for r in rows if r.kind == "mode" and r.status == "missing"]
    assert missing_modes
    for row in missing_modes:
        assert row.note in {"drop-out", "slice limit"}
    radar = [
        r for r in rows
        if r.program == "HYPER6" and r.kind == "vehicle" and r.name == "RADAR0"
    ]
    assert radar[0].status == "missing"
    sat3 = [
        r for r in rows
        if r.program == "HYPER6" and r.kind == "vehicle" and r.name == "SAT3"
    ]
    assert sat3[0].status == "missing"


def test_inventory_hyper6_seeker_name_match_only():
    rows = load_inventory(repo_root() / "Python/tools/cadac_cpp/inventory.json")
    seeker = [
        r for r in rows
        if r.program == "HYPER6" and r.kind == "module" and r.name == "seeker"
    ]
    assert len(seeker) == 1
    assert seeker[0].status == "ported"
    assert seeker[0].note == "name match only"
    assert "aim5" in (seeker[0].python or "")


from pathlib import Path

from cadac.cli import _VEHICLE_TYPES, _build_vehicle
from cadac.io.scenario import VehicleSpec, load_scenario


def test_hyper6_without_family_stays_hyper6():
    assert "HYPER6" in _VEHICLE_TYPES
    assert _VEHICLE_TYPES["HYPER6"].__name__ == "Hyper6"


def test_vehicle_families_registers_rocket6():
    import cadac.cli as cli
    from cadac.vehicles.round6.rocket6.vehicle import Rocket6

    assert hasattr(cli, "_VEHICLE_FAMILIES")
    assert cli._VEHICLE_FAMILIES.get(("rocket6", "HYPER6")) is Rocket6


def test_family_unknown_does_not_use_type_table(tmp_path: Path):
    spec = VehicleSpec(
        type="HYPER6",
        name="SLV",
        aero_deck=None,
        prop_deck=None,
        params={},
        events=[],
        family="no_such_family",
    )
    try:
        _build_vehicle(tmp_path / "x.jsonc", spec)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        msg = str(exc)
        assert "no_such_family" in msg
        assert "HYPER6" in msg
        assert "x.jsonc" in msg


def test_rocket6_without_aero_deck_does_not_require_prop(tmp_path: Path):
    spec = VehicleSpec(
        type="HYPER6",
        name="SLV",
        aero_deck=None,
        prop_deck=None,
        params={},
        events=[],
        family="rocket6",
    )
    try:
        _build_vehicle(tmp_path / "x.jsonc", spec)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        msg = str(exc)
        assert "aero_deck" in msg
        assert "and prop_deck" not in msg


def test_vehicle_family_wins_over_scenario(tmp_path: Path):
    path = tmp_path / "both.jsonc"
    path.write_text(
        '{"title":"t","options":{},"modules":[],"timing":{"int_step":0.01},'
        '"end_time":0,"family":"scenario_default",'
        '"vehicles":[{"type":"HYPER6","name":"SLV","family":"rocket6","params":{}}]}',
        encoding="utf-8",
        newline="\n",
    )
    v = load_scenario(path).vehicles[0]
    assert v.family == "rocket6"


def test_scenario_family_used_when_vehicle_omits_it(tmp_path: Path):
    path = tmp_path / "scen.jsonc"
    path.write_text(
        '{"title":"t","options":{},"modules":[],"timing":{"int_step":0.01},'
        '"end_time":0,"family":"rocket6",'
        '"vehicles":[{"type":"HYPER6","name":"SLV","params":{}}]}',
        encoding="utf-8",
        newline="\n",
    )
    assert load_scenario(path).vehicles[0].family == "rocket6"

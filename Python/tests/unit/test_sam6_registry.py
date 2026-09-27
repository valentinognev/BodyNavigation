from pathlib import Path

import pytest

from cadac.cli import (
    _VEHICLE_FAMILIES,
    _VEHICLE_TYPES,
    _build_vehicle,
    run_scenario,
)
from cadac.io.scenario import VehicleSpec
from cadac.vehicles.flat6.sam6.aircraft import Sam6Aircraft
from cadac.vehicles.flat6.sam6.environment import Sam6Environment
from cadac.vehicles.flat6.sam6.kinematics import Sam6Kinematics
from cadac.vehicles.flat6.sam6.radar import Sam6Radar
from cadac.vehicles.flat6.sam6.rocket import Sam6Rocket
from cadac.vehicles.flat6.sam6.vehicle import Sam6Missile


DUMMY_DECK = (
    '{ "title": "t", "tables": [ { "name": "dummy", "dim": 1, '
    '"x1": [0.0], "values": [0.0] } ] }'
)


def _dummy_deck(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text(DUMMY_DECK, encoding="utf-8", newline="\n")
    return path


def _missile_spec(tmp_path: Path, family="sam6"):
    return VehicleSpec(
        type="MISSILE6",
        name="SAM",
        aero_deck=_dummy_deck(tmp_path, "aero.jsonc"),
        prop_deck=_dummy_deck(tmp_path, "prop.jsonc"),
        params={},
        events=[],
        family=family,
    )


def test_build_vehicle_sam6_missile6_returns_sam6_missile(tmp_path: Path):
    vehicle = _build_vehicle(Path("x.jsonc"), _missile_spec(tmp_path))
    assert isinstance(vehicle, Sam6Missile)
    assert vehicle.type == "MISSILE6"


def test_family_none_missile6_raises(tmp_path: Path):
    spec = _missile_spec(tmp_path, family=None)
    with pytest.raises(ValueError, match="MISSILE6"):
        _build_vehicle(Path("x.jsonc"), spec)


def test_aircraft3_run_scenario_zero_s_does_not_require_aero(tmp_path: Path):
    path = tmp_path / "aircraft.jsonc"
    path.write_text(
        '{ "title": "t", "options": {}, "modules": [], '
        '"timing": { "int_step": 0.01 }, "end_time": 0, '
        '"vehicles": [ { "family": "sam6", "type": "AIRCRAFT3", '
        '"name": "a1", "params": {}, "events": [] } ] }',
        encoding="utf-8",
        newline="\n",
    )
    result = run_scenario(path)
    assert result.plot_rows is not None


def test_rocket5_without_aero_deck_raises():
    spec = VehicleSpec(
        type="ROCKET5",
        name="r1",
        aero_deck=None,
        prop_deck=None,
        params={},
        events=[],
        family="sam6",
    )
    with pytest.raises(ValueError, match="aero_deck"):
        _build_vehicle(Path("x.jsonc"), spec)


def test_radar0_mtrack_1_without_sam_deck_raises():
    spec = VehicleSpec(
        type="RADAR0",
        name="R",
        aero_deck=None,
        prop_deck=None,
        params={"mtrack": 1},
        events=[],
        family="sam6",
    )
    with pytest.raises(ValueError, match="sam_deck"):
        _build_vehicle(Path("x.jsonc"), spec)


def test_missile_modules_include_sam6_environment_and_kinematics(tmp_path: Path):
    vehicle = _build_vehicle(Path("x.jsonc"), _missile_spec(tmp_path))
    assert any(isinstance(module, Sam6Environment) for module in vehicle.modules)
    assert any(isinstance(module, Sam6Kinematics) for module in vehicle.modules)


def test_rocket5_with_prop_deck_raises(tmp_path: Path):
    spec = VehicleSpec(
        type="ROCKET5",
        name="r1",
        aero_deck=_dummy_deck(tmp_path, "aero.jsonc"),
        prop_deck=_dummy_deck(tmp_path, "prop.jsonc"),
        params={},
        events=[],
        family="sam6",
    )
    with pytest.raises(ValueError, match="prop_deck"):
        _build_vehicle(Path("x.jsonc"), spec)


def test_rocket5_with_aero_returns_sam6_rocket(tmp_path: Path):
    spec = VehicleSpec(
        type="ROCKET5",
        name="r1",
        aero_deck=_dummy_deck(tmp_path, "aero.jsonc"),
        prop_deck=None,
        params={},
        events=[],
        family="sam6",
    )
    vehicle = _build_vehicle(Path("x.jsonc"), spec)
    assert isinstance(vehicle, Sam6Rocket)
    assert vehicle.type == "ROCKET5"


def test_radar0_mtrack_0_without_traj_decks():
    spec = VehicleSpec(
        type="RADAR0",
        name="R",
        aero_deck=None,
        prop_deck=None,
        params={},
        events=[],
        family="sam6",
    )
    vehicle = _build_vehicle(Path("x.jsonc"), spec)
    assert isinstance(vehicle, Sam6Radar)
    assert vehicle.type == "RADAR0"


def test_sam6_types_are_family_only_not_global():
    for type_name, cls in (
        ("MISSILE6", Sam6Missile),
        ("AIRCRAFT3", Sam6Aircraft),
        ("ROCKET5", Sam6Rocket),
        ("RADAR0", Sam6Radar),
    ):
        assert type_name not in _VEHICLE_TYPES
        assert _VEHICLE_FAMILIES[("sam6", type_name)] is cls

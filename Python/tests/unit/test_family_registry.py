from pathlib import Path
import json
import pytest
from cadac.cli import _VEHICLE_TYPES, _VEHICLE_FAMILIES, register_family_type, _build_vehicle
from cadac.io.scenario import RunConfig, VehicleSpec, load_scenario
from cadac.io.translate import translate_scenario_asc


class _Dummy:
    type = "MISSILE6"


class _Other:
    type = "MISSILE6"


def _restore_families(before):
    _VEHICLE_FAMILIES.clear()
    _VEHICLE_FAMILIES.update(before)


def test_register_family_does_not_touch_global_types_or_wipe_map():
    before_types = dict(_VEHICLE_TYPES)
    before_fam = dict(_VEHICLE_FAMILIES)
    try:
        register_family_type("testfam", "MISSILE6", _Dummy)
        assert _VEHICLE_TYPES == before_types
        for key, cls in before_fam.items():
            assert _VEHICLE_FAMILIES[key] is cls  # AIM5 pairs stay
        register_family_type("testfam", "MISSILE6", _Dummy)  # idempotent
        assert _VEHICLE_FAMILIES[("testfam", "MISSILE6")] is _Dummy
    finally:
        _restore_families(before_fam)


def test_register_family_rejects_different_class_for_occupied_pair():
    before_fam = dict(_VEHICLE_FAMILIES)
    try:
        register_family_type("testfam", "MISSILE6", _Dummy)
        with pytest.raises(ValueError):
            register_family_type("testfam", "MISSILE6", _Other)
        assert _VEHICLE_FAMILIES[("testfam", "MISSILE6")] is _Dummy
    finally:
        _restore_families(before_fam)


def test_vehicle_family_none_when_omitted(tmp_path: Path):
    p = tmp_path / "s.jsonc"
    p.write_text(
        '{ "title": "t", "options": {}, "modules": [], '
        '"timing": {}, "end_time": 1, "vehicles": [ { "type": "CRUISE3", '
        '"name": "c", "params": {}, "events": [] } ] }',
        encoding="utf-8", newline="\n",
    )
    assert load_scenario(p).vehicles[0].family is None


def test_scenario_family_applies_when_vehicle_omits_it(tmp_path: Path):
    p = tmp_path / "s.jsonc"
    p.write_text(
        '{ "title": "t", "family": "sam6", "options": {}, "modules": [], '
        '"timing": {}, "end_time": 1, "vehicles": [ { "type": "CRUISE3", '
        '"name": "c", "params": {}, "events": [] } ] }',
        encoding="utf-8", newline="\n",
    )
    assert load_scenario(p).vehicles[0].family == "sam6"


def test_vehicle_family_overrides_scenario_family(tmp_path: Path):
    p = tmp_path / "s.jsonc"
    p.write_text(
        '{ "title": "t", "family": "sam6", "options": {}, "modules": [], '
        '"timing": {}, "end_time": 1, "vehicles": [ { "family": "aim5", '
        '"type": "CRUISE3", "name": "c", "params": {}, "events": [] } ] }',
        encoding="utf-8", newline="\n",
    )
    assert load_scenario(p).vehicles[0].family == "aim5"


def test_build_vehicle_reads_spec_family():
    spec = VehicleSpec(
        type="HYPER5", name="h", aero_deck=None, prop_deck=None,
        params={}, events=[], family="sam6",
    )
    with pytest.raises(ValueError, match="sam6") as excinfo:
        _build_vehicle(Path("x.jsonc"), spec)
    assert "HYPER5" in str(excinfo.value)


def test_load_traj_decks(tmp_path: Path):
    p = tmp_path / "s.jsonc"
    p.write_text(
        '{ "title": "t", "options": {}, "modules": [], '
        '"timing": {}, "end_time": 1, "vehicles": [ { "family": "sam6", '
        '"type": "RADAR0", "name": "R", "sam_deck": "SAM_traj_deck.jsonc", '
        '"srmb_deck": "SRBM_traj_deck_ballistic.jsonc", "params": {}, '
        '"events": [] } ] }',
        encoding="utf-8", newline="\n",
    )
    cfg = load_scenario(p)
    assert cfg.vehicles[0].family == "sam6"
    assert cfg.vehicles[0].sam_deck == tmp_path / "SAM_traj_deck.jsonc"
    assert cfg.vehicles[0].srmb_deck == tmp_path / "SRBM_traj_deck_ballistic.jsonc"
    # Source of truth is vehicles[0].family — do not require RunConfig.family
    assert "family" not in RunConfig.__dataclass_fields__


_TWO_VEHICLE_ASC = (
    "TITLE two-veh\n"
    "OPTIONS y_scrn\n"
    "MODULES\n"
    "\tenvironment\tdef,init,exec\n"
    "END\n"
    "TIMING\n"
    "\tint_step 0.01\n"
    "END\n"
    "VEHICLES 2\n"
    "\tMISSILE6 m1\n"
    "\t\tAERO_DECK  SAM_aero_deck.asc\n"
    "\t\tIF time > 5\n"
    "\t\t\tmaut  3\n"
    "\t\tENDIF\n"
    "\t\tENDIF\n"
    "\tEND\n"
    "\tRADAR0 R\n"
    "\t\tSAM_DECK    SAM_traj_deck.asc\n"
    "\t\tSRBM_DECK  SRBM_traj_deck_ballistic.asc\n"
    "\t\tsrel1  0\n"
    "\tEND\n"
    "END\n"
    "ENDTIME 1\n"
    "STOP\n"
)


def test_translate_stamps_family_on_each_vehicle_and_maps_traj_decks(tmp_path: Path):
    src = tmp_path / "two.asc"
    src.write_text(_TWO_VEHICLE_ASC, encoding="utf-8", newline="\n")
    translate_scenario_asc(src, tmp_path, family="sam6")
    data = json.loads((tmp_path / "two.jsonc").read_text(encoding="utf-8"))
    assert "family" not in data
    assert len(data["vehicles"]) == 2
    assert all(vehicle["family"] == "sam6" for vehicle in data["vehicles"])
    radar = data["vehicles"][1]
    assert radar["sam_deck"] == "SAM_traj_deck.jsonc"
    assert radar["srmb_deck"] == "SRBM_traj_deck_ballistic.jsonc"
    assert "SAM_DECK" not in radar["params"]
    assert "ENDIF" not in data["vehicles"][0]["params"]
    cfg = load_scenario(tmp_path / "two.jsonc")
    assert [v.family for v in cfg.vehicles] == ["sam6", "sam6"]
    assert cfg.vehicles[1].sam_deck == tmp_path / "SAM_traj_deck.jsonc"
    assert cfg.vehicles[1].srmb_deck == tmp_path / "SRBM_traj_deck_ballistic.jsonc"


def test_translate_omits_family_key_when_family_none(tmp_path: Path):
    src = tmp_path / "two.asc"
    src.write_text(_TWO_VEHICLE_ASC, encoding="utf-8", newline="\n")
    translate_scenario_asc(src, tmp_path)
    data = json.loads((tmp_path / "two.jsonc").read_text(encoding="utf-8"))
    assert "family" not in data
    for vehicle in data["vehicles"]:
        assert "family" not in vehicle

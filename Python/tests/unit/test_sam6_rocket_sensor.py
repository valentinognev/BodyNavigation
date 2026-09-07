from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.sam6.rocket import Sam6RocketSensor

RTOL = 1e-12
ATOL = 1e-14
SMALL = 1e-7
ZEROS3 = (0.0, 0.0, 0.0)
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

STEL1 = 10000.0
STEL2 = 2000.0
STEL3 = -500.0
SAEL = np.array([0.0, 0.0, -8000.0], dtype=float)
VAEL = np.array([200.0, -50.0, 30.0], dtype=float)
ALT = 1000.0
ALT_ENDO = 30000.0
SENTINEL_DTA = 777.0

DEFINED = (
    "mseek",
    "stel1",
    "stel2",
    "stel3",
    "dta",
    "dvta",
    "tgo_tgt",
    "los_azx",
    "los_elx",
    "sigdy",
    "sigdz",
    "UTAA",
    "WOEA",
    "STAL",
)
ROLES = {
    "mseek": "data",
    "stel1": "data",
    "stel2": "data",
    "stel3": "data",
    "dta": "out",
    "dvta": "out",
    "tgo_tgt": "diag",
    "los_azx": "diag",
    "los_elx": "diag",
    "sigdy": "diag",
    "sigdz": "diag",
    "UTAA": "out",
    "WOEA": "out",
    "STAL": "out",
}
OUTPUTS = {
    "mseek": (),
    "stel1": (),
    "stel2": (),
    "stel3": (),
    "dta": ("com",),
    "dvta": ("com",),
    "tgo_tgt": (),
    "los_azx": (),
    "los_elx": (),
    "sigdy": (),
    "sigdz": (),
    "UTAA": (),
    "WOEA": (),
    "STAL": (),
}
TYPES = {
    "mseek": "int",
    "stel1": "real",
    "stel2": "real",
    "stel3": "real",
    "dta": "real",
    "dvta": "real",
    "tgo_tgt": "real",
    "los_azx": "real",
    "los_elx": "real",
    "sigdy": "real",
    "sigdz": "real",
    "UTAA": "vec",
    "WOEA": "vec",
    "STAL": "vec",
}
DEFAULTS = {
    "mseek": 0,
    "stel1": 0.0,
    "stel2": 0.0,
    "stel3": 0.0,
    "dta": 0.0,
    "dvta": 0.0,
    "tgo_tgt": 9999.0,
    "los_azx": 0.0,
    "los_elx": 0.0,
    "sigdy": 0.0,
    "sigdz": 0.0,
}
NOT_DEFINED = (
    "TAL",
    "SAEL",
    "VAEL",
    "alt",
    "flag_exo",
    "alt_endo",
    "time",
    "STEL",
    "VTEL",
    "mtarget",
    "SBEL",
    "TBL",
    "ancomx",
    "alcomx",
    "maut",
    "mguide",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.001,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _skew(vec):
    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def _expected_kin(stel, sael, vael, tal):
    stal = stel - sael
    dta = float(np.linalg.norm(stal))
    if dta == 0.0:
        utal = np.zeros(3, dtype=float)
    else:
        utal = stal / dta
    utaa = tal @ utal
    vtael = vael * (-1.0)
    dvta = float(utal @ vtael)
    abs_dvta = abs(dvta)
    tgo_tgt = dta / abs_dvta if abs_dvta > SMALL else 0.0
    woea = tal @ _skew(utal) @ vtael * (1.0 / dta)
    return {
        "dta": dta,
        "dvta": dvta,
        "tgo_tgt": tgo_tgt,
        "UTAA": utaa,
        "WOEA": woea,
        "STAL": stal,
        "sigdy": float(woea[1]),
        "sigdz": float(woea[2]),
    }


def _plant(
    store,
    *,
    sael=SAEL,
    vael=VAEL,
    tal=IDENTITY,
    alt=ALT,
    flag_exo=1,
    alt_endo=ALT_ENDO,
):
    store.define(Field("TAL", tal, "mat", "out", "kinematics"))
    store.define(Field("SAEL", sael, "vec", "state", "newton"))
    store.define(Field("VAEL", vael, "vec", "out", "newton"))
    store.define(Field("alt", alt, "real", "out", "newton"))
    store.define(Field("flag_exo", flag_exo, "int", "data", "control"))
    store.define(Field("alt_endo", alt_endo, "real", "data", "control"))


def _defined():
    vehicle = _Vehicle()
    sensor = Sam6RocketSensor()
    sensor.define(vehicle)
    return vehicle, sensor


def _ready(
    *,
    mseek=1,
    stel1=STEL1,
    stel2=STEL2,
    stel3=STEL3,
    sael=SAEL,
    vael=VAEL,
    tal=IDENTITY,
    alt=ALT,
    flag_exo=1,
    alt_endo=ALT_ENDO,
):
    vehicle, sensor = _defined()
    store = vehicle.store
    store.set("mseek", mseek)
    store.set("stel1", stel1)
    store.set("stel2", stel2)
    store.set("stel3", stel3)
    _plant(
        store,
        sael=sael,
        vael=vael,
        tal=tal,
        alt=alt,
        flag_exo=flag_exo,
        alt_endo=alt_endo,
    )
    return vehicle, sensor


def test_name_is_sensor():
    assert Sam6RocketSensor.name == "sensor"


def test_define_cpp_fields():
    vehicle, _ = _defined()
    store = vehicle.store
    assert tuple(store.names()) == DEFINED
    for name in DEFINED:
        field = store.field(name)
        assert field.module == "sensor", name
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        assert field.type == TYPES[name], name
    for name, value in DEFAULTS.items():
        assert _approx(store.get(name), value), name
    np.testing.assert_allclose(store.get("UTAA"), ZEROS3, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("WOEA"), ZEROS3, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("STAL"), ZEROS3, rtol=RTOL, atol=ATOL)
    assert store.get("UTAA").shape == (3,)
    assert store.get("WOEA").shape == (3,)
    assert store.get("STAL").shape == (3,)


def test_define_does_not_register_flat3_or_control():
    vehicle, _ = _defined()
    names = vehicle.store.names()
    for name in NOT_DEFINED:
        assert name not in names


def test_mseek_0_returns_without_raise_or_dta_write():
    vehicle, sensor = _ready(mseek=0)
    store = vehicle.store
    store.set("dta", SENTINEL_DTA)
    sensor.execute(vehicle, _ctx())
    assert store.get("mseek") == 0
    assert _approx(store.get("dta"), SENTINEL_DTA)


def test_mseek_1_flag_exo_alt_below_endo_dta_finite():
    vehicle, sensor = _ready(mseek=1, flag_exo=1, alt=ALT, alt_endo=ALT_ENDO)
    sensor.execute(vehicle, _ctx())
    dta = vehicle.store.get("dta")
    assert np.isfinite(dta)
    stel = np.array([STEL1, STEL2, STEL3], dtype=float)
    want = float(np.linalg.norm(stel - SAEL))
    assert _approx(dta, want)
    assert want > 0.0


def test_kinematic_los_matches_cpp_replica():
    tal = np.array(IDENTITY, dtype=float)
    vehicle, sensor = _ready(
        mseek=1,
        stel1=STEL1,
        stel2=STEL2,
        stel3=STEL3,
        sael=SAEL,
        vael=VAEL,
        tal=tal,
        flag_exo=1,
        alt=ALT,
        alt_endo=ALT_ENDO,
    )
    sensor.execute(vehicle, _ctx())
    stel = np.array([STEL1, STEL2, STEL3], dtype=float)
    want = _expected_kin(stel, SAEL, VAEL, tal)
    store = vehicle.store
    assert _approx(store.get("dta"), want["dta"])
    assert _approx(store.get("dvta"), want["dvta"])
    assert _approx(store.get("tgo_tgt"), want["tgo_tgt"])
    assert _approx(store.get("sigdy"), want["sigdy"])
    assert _approx(store.get("sigdz"), want["sigdz"])
    np.testing.assert_allclose(store.get("STAL"), want["STAL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("UTAA"), want["UTAA"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("WOEA"), want["WOEA"], rtol=RTOL, atol=ATOL)
    dta_hand = sqrt(STEL1 ** 2 + STEL2 ** 2 + (STEL3 - SAEL[2]) ** 2)
    assert _approx(store.get("dta"), dta_hand)


def test_mseek_nonzero_conditions_off_does_nothing():
    vehicle, sensor = _ready(mseek=1, flag_exo=0, alt=ALT, alt_endo=ALT_ENDO)
    store = vehicle.store
    store.set("dta", SENTINEL_DTA)
    store.set("dvta", 12.0)
    store.set("tgo_tgt", 34.0)
    store.set("sigdy", 1.0)
    store.set("sigdz", 2.0)
    store.set("UTAA", (1.0, 2.0, 3.0))
    store.set("WOEA", (4.0, 5.0, 6.0))
    store.set("STAL", (7.0, 8.0, 9.0))
    sensor.execute(vehicle, _ctx())
    assert store.get("mseek") == 1
    assert _approx(store.get("dta"), SENTINEL_DTA)
    assert _approx(store.get("dvta"), 12.0)
    assert _approx(store.get("tgo_tgt"), 34.0)
    assert _approx(store.get("sigdy"), 1.0)
    assert _approx(store.get("sigdz"), 2.0)
    np.testing.assert_allclose(store.get("UTAA"), (1.0, 2.0, 3.0), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("WOEA"), (4.0, 5.0, 6.0), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("STAL"), (7.0, 8.0, 9.0), rtol=RTOL, atol=ATOL)


def test_mseek_nonzero_alt_not_below_endo_does_nothing():
    vehicle, sensor = _ready(mseek=1, flag_exo=1, alt=ALT_ENDO, alt_endo=ALT_ENDO)
    store = vehicle.store
    store.set("dta", SENTINEL_DTA)
    sensor.execute(vehicle, _ctx())
    assert _approx(store.get("dta"), SENTINEL_DTA)


def test_mseek_2_truthy_still_tracks():
    vehicle, sensor = _ready(mseek=2, flag_exo=1, alt=ALT, alt_endo=ALT_ENDO)
    sensor.execute(vehicle, _ctx())
    assert np.isfinite(vehicle.store.get("dta"))
    assert vehicle.store.get("dta") > 0.0


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.sam6.rocket as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "Plane5" not in src
    assert "Plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src


def test_initialize_and_terminate_are_pass():
    vehicle, sensor = _ready()
    ctx = _ctx()
    assert sensor.initialize(vehicle, ctx) is None
    sensor.execute(vehicle, ctx)
    assert sensor.terminate(vehicle, ctx) is None

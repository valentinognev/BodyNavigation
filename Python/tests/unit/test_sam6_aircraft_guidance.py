from pathlib import Path

import numpy as np
import pytest

from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.sam6.aircraft import Sam6AircraftGuidance

RTOL = 1e-12
ATOL = 1e-14
GRAV = 9.8
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
YAW90 = ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
ZEROS3 = (0.0, 0.0, 0.0)

DEFINED = (
    "acft_option",
    "guid_gain",
    "ACOML",
    "gturn",
    "man_start",
    "man_stop",
)
ROLES = {
    "acft_option": "data",
    "guid_gain": "data",
    "ACOML": "out",
    "gturn": "data",
    "man_start": "data",
    "man_stop": "data",
}
OUTPUTS = {
    "acft_option": (),
    "guid_gain": (),
    "ACOML": (),
    "gturn": (),
    "man_start": (),
    "man_stop": (),
}
INT_FIELDS = ("acft_option",)
VEC_FIELDS = ("ACOML",)
NOT_DEFINED = (
    "time",
    "grav",
    "TVL",
    "SAEL",
    "VAEL",
    "phiav",
    "phiavd",
    "anx",
    "anxd",
    "FSPA",
    "phiavout",
    "acc_longx",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ctx(sim_time=0.0, combus=None, vehicle_slot=0):
    return SimContext(
        sim_time=sim_time,
        int_step=0.001,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=vehicle_slot,
    )


def _plant(
    store,
    *,
    time=0.0,
    grav=GRAV,
    tvl=IDENTITY,
    sael=(0.0, 0.0, 0.0),
    vael=(0.0, 250.0, 0.0),
):
    store.define(Field("time", time, "real", "exec", "kinematics"))
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("TVL", tvl, "mat", "out", "newton"))
    store.define(Field("SAEL", sael, "vec", "out", "newton"))
    store.define(Field("VAEL", vael, "vec", "out", "newton"))


def _ready(
    *,
    acft_option=0,
    guid_gain=0.0,
    gturn=0.0,
    man_start=0.0,
    man_stop=0.0,
    time=0.0,
    grav=GRAV,
    tvl=IDENTITY,
    sael=(0.0, 0.0, 0.0),
    vael=(0.0, 250.0, 0.0),
):
    vehicle = _Vehicle()
    guidance = Sam6AircraftGuidance()
    guidance.define(vehicle)
    guidance.initialize(vehicle, _ctx(sim_time=time))
    store = vehicle.store
    store.set("acft_option", acft_option)
    store.set("guid_gain", guid_gain)
    store.set("gturn", gturn)
    store.set("man_start", man_start)
    store.set("man_stop", man_stop)
    _plant(store, time=time, grav=grav, tvl=tvl, sael=sael, vael=vael)
    return vehicle, guidance


def _missile(name, sbel, vbel):
    return Packet(
        name=name,
        type="MISSILE6",
        status=1,
        vars={
            "SBEL": np.array(sbel, dtype=float),
            "VBEL": np.array(vbel, dtype=float),
        },
    )


def test_name_is_guidance():
    assert Sam6AircraftGuidance().name == "guidance"


def test_define_registers_cpp_fields():
    vehicle = _Vehicle()
    Sam6AircraftGuidance().define(vehicle)
    store = vehicle.store
    zeros3 = np.zeros(3)
    assert tuple(store.names()) == DEFINED
    for name in DEFINED:
        field = store.field(name)
        assert field.module == "guidance", name
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        if name in INT_FIELDS:
            assert field.type == "int", name
            assert store.get(name) == 0, name
        elif name in VEC_FIELDS:
            assert field.type == "vec", name
            np.testing.assert_array_equal(store.get(name), zeros3)
            assert store.get(name).shape == (3,)
        else:
            assert field.type == "real", name
            assert _approx(store.get(name), 0.0), name
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_initialize_is_pass():
    vehicle = _Vehicle()
    guidance = Sam6AircraftGuidance()
    guidance.define(vehicle)
    vehicle.store.set("ACOML", (1.0, 2.0, 3.0))
    vehicle.store.set("gturn", 4.0)
    before_acoml = np.array(vehicle.store.get("ACOML"), copy=True)
    before_gturn = vehicle.store.get("gturn")
    assert guidance.initialize(vehicle, _ctx()) is None
    np.testing.assert_allclose(
        vehicle.store.get("ACOML"), before_acoml, rtol=RTOL, atol=ATOL
    )
    assert _approx(vehicle.store.get("gturn"), before_gturn)


def test_option_zero_gravity_bias():
    vehicle, guidance = _ready(acft_option=0, grav=GRAV)
    guidance.execute(vehicle, _ctx())
    np.testing.assert_allclose(
        vehicle.store.get("ACOML"), (0.0, 0.0, -GRAV), rtol=RTOL, atol=ATOL
    )


def test_option_zero_gravity_inside_maneuver_window():
    vehicle, guidance = _ready(
        acft_option=0,
        gturn=1.0,
        man_start=0.0,
        man_stop=10.0,
        time=1.0,
        grav=GRAV,
    )
    guidance.execute(vehicle, _ctx(sim_time=1.0))
    np.testing.assert_allclose(
        vehicle.store.get("ACOML"), (0.0, 0.0, -GRAV), rtol=RTOL, atol=ATOL
    )


def test_option_three_raises():
    vehicle, guidance = _ready(acft_option=3)
    vehicle.store.set("ACOML", (1.0, 2.0, 3.0))
    with pytest.raises(ValueError):
        guidance.execute(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("ACOML"), (1.0, 2.0, 3.0))


@pytest.mark.parametrize("acft_option", [-1, 4, 10])
def test_unknown_option_raises(acft_option):
    vehicle, guidance = _ready(acft_option=acft_option)
    with pytest.raises(ValueError):
        guidance.execute(vehicle, _ctx())


def test_option_one_horizontal_gturn_identity_tvl():
    vehicle, guidance = _ready(
        acft_option=1,
        gturn=1.0,
        man_start=0.0,
        man_stop=10.0,
        time=1.0,
        tvl=IDENTITY,
        grav=GRAV,
    )
    guidance.execute(vehicle, _ctx(sim_time=1.0))
    acoml = vehicle.store.get("ACOML")
    assert _approx(acoml[1], GRAV)
    np.testing.assert_allclose(acoml, (0.0, GRAV, -GRAV), rtol=RTOL, atol=ATOL)


def test_option_one_uses_tvl_transpose():
    vehicle, guidance = _ready(
        acft_option=1,
        gturn=1.0,
        man_start=0.0,
        man_stop=10.0,
        time=1.0,
        tvl=YAW90,
        grav=GRAV,
    )
    guidance.execute(vehicle, _ctx(sim_time=1.0))
    np.testing.assert_allclose(
        vehicle.store.get("ACOML"), (-GRAV, 0.0, -GRAV), rtol=RTOL, atol=ATOL
    )


def test_option_one_inclusive_man_start():
    vehicle, guidance = _ready(
        acft_option=1,
        gturn=1.0,
        man_start=10.0,
        man_stop=20.0,
        time=10.0,
        tvl=IDENTITY,
        grav=GRAV,
    )
    guidance.execute(vehicle, _ctx(sim_time=10.0))
    assert _approx(vehicle.store.get("ACOML")[1], GRAV)


def test_option_one_exclusive_man_stop_gravity_bias():
    vehicle, guidance = _ready(
        acft_option=1,
        gturn=1.0,
        man_start=10.0,
        man_stop=20.0,
        time=20.0,
        tvl=IDENTITY,
        grav=GRAV,
    )
    guidance.execute(vehicle, _ctx(sim_time=20.0))
    np.testing.assert_allclose(
        vehicle.store.get("ACOML"), (0.0, 0.0, -GRAV), rtol=RTOL, atol=ATOL
    )


def test_option_one_before_window_gravity_bias():
    vehicle, guidance = _ready(
        acft_option=1,
        gturn=1.0,
        man_start=10.0,
        man_stop=20.0,
        time=9.0,
        tvl=IDENTITY,
        grav=GRAV,
    )
    guidance.execute(vehicle, _ctx(sim_time=9.0))
    np.testing.assert_allclose(
        vehicle.store.get("ACOML"), (0.0, 0.0, -GRAV), rtol=RTOL, atol=ATOL
    )


def test_option_two_escape_vs_first_missile6():
    missile = _missile("m1", sbel=(0.0, 0.0, 0.0), vbel=(16.0, 0.0, 0.0))
    vehicle, guidance = _ready(
        acft_option=2,
        guid_gain=1.0,
        man_start=0.0,
        man_stop=10.0,
        time=1.0,
        grav=GRAV,
        sael=(1000.0, 0.0, 0.0),
        vael=(0.0, 250.0, 0.0),
    )
    guidance.execute(vehicle, _ctx(sim_time=1.0, combus=[missile]))
    np.testing.assert_allclose(
        vehicle.store.get("ACOML"), (4.0, 0.0, -GRAV), rtol=RTOL, atol=ATOL
    )


def test_option_two_uses_first_missile6_by_type_not_name():
    decoy = Packet(
        name="m1",
        type="ROCKET5",
        status=1,
        vars={
            "SBEL": np.array([500.0, 0.0, 0.0], dtype=float),
            "VBEL": np.array([0.0, 16.0, 0.0], dtype=float),
        },
    )
    later = _missile("m2", sbel=(200.0, 0.0, 0.0), vbel=(0.0, 16.0, 0.0))
    first = _missile("not-m1", sbel=(0.0, 0.0, 0.0), vbel=(16.0, 0.0, 0.0))
    vehicle, guidance = _ready(
        acft_option=2,
        guid_gain=1.0,
        man_start=0.0,
        man_stop=10.0,
        time=1.0,
        grav=GRAV,
        sael=(1000.0, 0.0, 0.0),
        vael=(0.0, 250.0, 0.0),
    )
    guidance.execute(
        vehicle,
        _ctx(sim_time=1.0, combus=[decoy, first, later], vehicle_slot=0),
    )
    np.testing.assert_allclose(
        vehicle.store.get("ACOML"), (4.0, 0.0, -GRAV), rtol=RTOL, atol=ATOL
    )


def test_option_two_outside_window_gravity_bias():
    missile = _missile("m1", sbel=(0.0, 0.0, 0.0), vbel=(16.0, 0.0, 0.0))
    vehicle, guidance = _ready(
        acft_option=2,
        guid_gain=1.0,
        man_start=10.0,
        man_stop=20.0,
        time=5.0,
        grav=GRAV,
        sael=(1000.0, 0.0, 0.0),
        vael=(0.0, 250.0, 0.0),
    )
    guidance.execute(vehicle, _ctx(sim_time=5.0, combus=[missile]))
    np.testing.assert_allclose(
        vehicle.store.get("ACOML"), (0.0, 0.0, -GRAV), rtol=RTOL, atol=ATOL
    )


def test_option_two_missing_missile6_raises():
    vehicle, guidance = _ready(
        acft_option=2,
        guid_gain=1.0,
        man_start=0.0,
        man_stop=10.0,
        time=1.0,
        grav=GRAV,
        sael=(1000.0, 0.0, 0.0),
        vael=(0.0, 250.0, 0.0),
    )
    with pytest.raises(ValueError):
        guidance.execute(vehicle, _ctx(sim_time=1.0, combus=[]))


def test_terminate_is_pass():
    vehicle, guidance = _ready(acft_option=0)
    assert guidance.terminate(vehicle, _ctx()) is None
    np.testing.assert_allclose(
        vehicle.store.get("ACOML"), np.zeros(3), rtol=RTOL, atol=ATOL
    )


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.flat6.sam6.aircraft as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "Plane5" not in src
    assert "Plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src

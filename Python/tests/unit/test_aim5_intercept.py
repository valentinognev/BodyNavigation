import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat2tr, polar_from_cart
from cadac.vehicles.flat3.aim5.intercept import Aim5Intercept

RTOL = 1e-12
ATOL = 1e-14

TIME = 8.0
VBEL = np.array([200.0, 50.0, -10.0])
VTEL = np.array([250.0, -20.0, 0.0])
PSIVLX_ACFT = -90.0
THTVLX_ACFT = 5.0
STAL = np.array([50.0, 80.0, -10.0])
ACFT_COM_SLOT = 1
VEHICLE_SLOT = 0

INTERCEPT_FIELDS = {
    "aspazx": ("real", "diag", 0.0, ()),
    "aspelx": ("real", "diag", 0.0, ()),
}

NOT_DEFINED = (
    "halt",
    "write",
    "miss",
    "hit_time",
    "MISS_G",
    "stop_run",
    "acft_com_slot",
    "VTEL",
    "psivlx_acft",
    "thtvlx_acft",
    "dta",
    "dvta",
    "STAL",
    "VBEL",
    "time",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()
        self.health = 1


def _expected_aspect(vtel, vbel, psivlx_acft, thtvlx_acft):
    vtael = vtel - vbel
    ttl = mat2tr(psivlx_acft * RAD, thtvlx_acft * RAD)
    polar = polar_from_cart(ttl @ vtael)
    return polar[1] * DEG, polar[2] * DEG


def _ctx(combus=None, vehicle_slot=VEHICLE_SLOT):
    if combus is None:
        combus = [
            Packet(name="Missile", type="AIM5", status=1, vars={}),
            Packet(name="Target", type="AIRCRAFT3", status=1, vars={}),
        ]
    return SimContext(
        sim_time=TIME,
        int_step=0.002,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=vehicle_slot,
    )


def _ready(*, dta, dvta, acft_com_slot=ACFT_COM_SLOT):
    vehicle = _Vehicle()
    intercept = Aim5Intercept()
    intercept.define(vehicle)
    store = vehicle.store
    store.define(Field("time", TIME, "real", "exec", "environment"))
    store.define(Field("VBEL", VBEL, "vec", "state", "newton"))
    store.define(Field("acft_com_slot", acft_com_slot, "int", "out", "combus"))
    store.define(Field("VTEL", VTEL, "vec", "out", "combus"))
    store.define(Field("psivlx_acft", PSIVLX_ACFT, "real", "out", "combus"))
    store.define(Field("thtvlx_acft", THTVLX_ACFT, "real", "out", "combus"))
    store.define(Field("dta", dta, "real", "out", "seeker"))
    store.define(Field("dvta", dvta, "real", "out", "seeker"))
    store.define(Field("STAL", STAL, "vec", "out", "seeker"))
    intercept.initialize(vehicle, _ctx())
    return vehicle, intercept, _ctx()


def test_name_is_intercept():
    assert Aim5Intercept().name == "intercept"


def test_define_registers_aspazx_aspelx_diag():
    vehicle = _Vehicle()
    Aim5Intercept().define(vehicle)
    store = vehicle.store
    assert list(INTERCEPT_FIELDS) == store.names()
    for name, (ftype, role, default, outputs) in INTERCEPT_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "intercept"
        assert field.outputs == outputs
        assert store.get(name) == pytest.approx(default, abs=ATOL)


def test_define_does_not_register_halt_write_ground_or_seeker():
    vehicle = _Vehicle()
    Aim5Intercept().define(vehicle)
    for name in NOT_DEFINED:
        with pytest.raises(KeyError):
            vehicle.store.get(name)


def test_initialize_and_terminate_are_pass():
    vehicle = _Vehicle()
    intercept = Aim5Intercept()
    intercept.define(vehicle)
    intercept.initialize(vehicle, _ctx())
    intercept.terminate(vehicle, _ctx())
    assert vehicle.health == 1
    assert vehicle.store.get("aspazx") == pytest.approx(0.0, abs=ATOL)
    assert vehicle.store.get("aspelx") == pytest.approx(0.0, abs=ATOL)


def test_dta_100_dvta_50_kills_and_matches_aspect_replica():
    vehicle, intercept, ctx = _ready(dta=100.0, dvta=50.0)
    want_az, want_el = _expected_aspect(VTEL, VBEL, PSIVLX_ACFT, THTVLX_ACFT)

    intercept.execute(vehicle, ctx)

    assert vehicle.health == 0
    assert ctx.combus[ctx.vehicle_slot].status == 0
    assert ctx.combus[ACFT_COM_SLOT].status == 0
    assert vehicle.store.get("aspazx") == pytest.approx(want_az, rel=RTOL, abs=ATOL)
    assert vehicle.store.get("aspelx") == pytest.approx(want_el, rel=RTOL, abs=ATOL)


def test_dta_100_dvta_negative_still_closing_health_stays_1():
    vehicle, intercept, ctx = _ready(dta=100.0, dvta=-50.0)

    intercept.execute(vehicle, ctx)

    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert ctx.combus[ACFT_COM_SLOT].status == 1
    assert vehicle.store.get("aspazx") == pytest.approx(0.0, abs=ATOL)
    assert vehicle.store.get("aspelx") == pytest.approx(0.0, abs=ATOL)


def test_dta_600_dvta_50_health_stays_1():
    vehicle, intercept, ctx = _ready(dta=600.0, dvta=50.0)

    intercept.execute(vehicle, ctx)

    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert ctx.combus[ACFT_COM_SLOT].status == 1
    assert vehicle.store.get("aspazx") == pytest.approx(0.0, abs=ATOL)
    assert vehicle.store.get("aspelx") == pytest.approx(0.0, abs=ATOL)


def test_dta_500_does_not_kill():
    vehicle, intercept, ctx = _ready(dta=500.0, dvta=50.0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert ctx.combus[ACFT_COM_SLOT].status == 1


def test_dvta_zero_inside_sphere_does_not_kill():
    vehicle, intercept, ctx = _ready(dta=100.0, dvta=0.0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 1
    assert ctx.combus[ctx.vehicle_slot].status == 1
    assert ctx.combus[ACFT_COM_SLOT].status == 1


def test_does_not_sys_exit_on_kill():
    vehicle, intercept, ctx = _ready(dta=100.0, dvta=50.0)
    intercept.execute(vehicle, ctx)
    assert vehicle.health == 0

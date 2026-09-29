"""HYPER6 RCS — Task 12: mrcs_moment type 1/2, mrcs_force 1."""

import numpy as np
import pytest

from cadac.constants import AGRAV, DEG
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round6.hyper6.rcs import Hyper6Rcs, rcs_prop, rcs_schmitt
from cadac.vehicles.round6.hyper6.vehicle import Hyper6

RTOL = 1e-12
ATOL = 1e-14

DEAD_ZONE = 0.4
HYSTERESIS = 0.1
RCS_TAU = 1.0
ROLL_MOM_MAX = 500.0
PITCH_MOM_MAX = 15000.0
YAW_MOM_MAX = 5000.0
RCS_ZETA = 0.7
RCS_FREQ = 1.0
SIDE_FORCE_MAX = 10000.0
THTBDCOMX = 50.0
PSIBCOMX = 50.0
VMASS = 1000.0
IBBB_DIAG = (2.0, 40.0, 45.0)
IBBB = np.diag(IBBB_DIAG)

DEFINED = (
    "mrcs_moment",
    "mrcs_force",
    "dead_zone",
    "hysteresis",
    "rcs_tau",
    "roll_mom_max",
    "pitch_mom_max",
    "yaw_mom_max",
    "rcs_zeta",
    "rcs_freq",
    "roll_save",
    "pitch_save",
    "yaw_save",
    "FMRCS",
    "phibdcomx",
    "thtbdcomx",
    "psibdcomx",
    "e_roll",
    "e_pitch",
    "e_yaw",
    "o_roll",
    "o_pitch",
    "o_yaw",
    "roll_count",
    "pitch_count",
    "yaw_count",
    "side_force_max",
    "FARCS",
    "e_right",
    "e_down",
    "o_right",
    "o_down",
    "right_save",
    "down_save",
    "rcs_minit_flag",
)

ROLES = {
    "mrcs_moment": "data",
    "mrcs_force": "data",
    "dead_zone": "data",
    "hysteresis": "data",
    "rcs_tau": "data",
    "roll_mom_max": "data",
    "pitch_mom_max": "data",
    "yaw_mom_max": "data",
    "rcs_zeta": "data",
    "rcs_freq": "data",
    "roll_save": "save",
    "pitch_save": "save",
    "yaw_save": "save",
    "FMRCS": "out",
    "phibdcomx": "data",
    "thtbdcomx": "data",
    "psibdcomx": "data",
    "e_roll": "diag",
    "e_pitch": "diag",
    "e_yaw": "diag",
    "o_roll": "save",
    "o_pitch": "save",
    "o_yaw": "save",
    "roll_count": "save",
    "pitch_count": "save",
    "yaw_count": "save",
    "side_force_max": "data",
    "FARCS": "out",
    "e_right": "diag",
    "e_down": "diag",
    "o_right": "save",
    "o_down": "save",
    "right_save": "save",
    "down_save": "save",
    "rcs_minit_flag": "init",
}

OUTPUTS = {
    "mrcs_moment": (),
    "mrcs_force": (),
    "dead_zone": (),
    "hysteresis": (),
    "rcs_tau": (),
    "roll_mom_max": (),
    "pitch_mom_max": (),
    "yaw_mom_max": (),
    "rcs_zeta": (),
    "rcs_freq": (),
    "roll_save": (),
    "pitch_save": (),
    "yaw_save": (),
    "FMRCS": (),
    "phibdcomx": (),
    "thtbdcomx": (),
    "psibdcomx": (),
    "e_roll": (),
    "e_pitch": (),
    "e_yaw": (),
    "o_roll": (),
    "o_pitch": (),
    "o_yaw": (),
    "roll_count": ("plot",),
    "pitch_count": ("plot",),
    "yaw_count": ("plot",),
    "side_force_max": (),
    "FARCS": (),
    "e_right": (),
    "e_down": (),
    "o_right": (),
    "o_down": (),
    "right_save": (),
    "down_save": (),
    "rcs_minit_flag": (),
}

INT_FIELDS = (
    "mrcs_moment",
    "mrcs_force",
    "o_roll",
    "o_pitch",
    "o_yaw",
    "roll_count",
    "pitch_count",
    "yaw_count",
    "o_right",
    "o_down",
    "rcs_minit_flag",
)
VEC_FIELDS = ("FMRCS", "FARCS")
NOT_DEFINED = (
    "phibdcx",
    "thtbdcx",
    "psibdcx",
    "psivdcx",
    "UTBC",
    "ppcx",
    "qqcx",
    "rrcx",
    "FSPCB",
    "IBBB",
    "vmass",
    "aycomx",
    "azcomx",
    "minit",
    "beco_flag",
    "acc_gain",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(dt=0.001):
    return SimContext(
        sim_time=0.0,
        int_step=dt,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _plant_externals(
    store,
    *,
    ppcx=0.0,
    qqcx=0.0,
    rrcx=0.0,
    phibdcx=0.0,
    thtbdcx=0.0,
    psibdcx=0.0,
    psivdcx=0.0,
    utbc=(0.0, 0.0, 0.0),
    ibbb=None,
    fspcb=(0.0, 0.0, 0.0),
    aycomx=0.0,
    azcomx=0.0,
    vmass=VMASS,
    minit=0,
    beco_flag=0,
):
    store.define(Field("ppcx", ppcx, "real", "out", "ins"))
    store.define(Field("qqcx", qqcx, "real", "out", "ins"))
    store.define(Field("rrcx", rrcx, "real", "out", "ins"))
    store.define(Field("phibdcx", phibdcx, "real", "out", "ins"))
    store.define(Field("thtbdcx", thtbdcx, "real", "out", "ins"))
    store.define(Field("psibdcx", psibdcx, "real", "out", "ins"))
    store.define(Field("psivdcx", psivdcx, "real", "out", "ins"))
    store.define(Field("UTBC", utbc, "vec", "out", "guidance"))
    if ibbb is None:
        ibbb = IBBB
    store.define(Field("IBBB", ibbb, "mat", "out", "propulsion"))
    store.define(Field("FSPCB", fspcb, "vec", "out", "ins"))
    store.define(Field("aycomx", aycomx, "real", "out", "guidance"))
    store.define(Field("azcomx", azcomx, "real", "out", "guidance"))
    store.define(Field("vmass", vmass, "real", "out", "propulsion"))
    store.define(Field("minit", minit, "int", "data", "newton"))
    store.define(Field("beco_flag", beco_flag, "int", "diag", "guidance"))


def _ready(
    *,
    mrcs_moment=21,
    mrcs_force=0,
    dead_zone=DEAD_ZONE,
    hysteresis=HYSTERESIS,
    rcs_tau=RCS_TAU,
    roll_mom_max=ROLL_MOM_MAX,
    pitch_mom_max=PITCH_MOM_MAX,
    yaw_mom_max=YAW_MOM_MAX,
    rcs_zeta=RCS_ZETA,
    rcs_freq=RCS_FREQ,
    side_force_max=SIDE_FORCE_MAX,
    phibdcomx=0.0,
    thtbdcomx=THTBDCOMX,
    psibdcomx=PSIBCOMX,
    ppcx=0.0,
    qqcx=0.0,
    rrcx=0.0,
    phibdcx=0.0,
    thtbdcx=0.0,
    psibdcx=0.0,
    psivdcx=0.0,
    utbc=(0.0, 0.0, 0.0),
    ibbb=None,
    fspcb=(0.0, 0.0, 0.0),
    aycomx=0.0,
    azcomx=0.0,
    vmass=VMASS,
    minit=0,
    beco_flag=0,
    **states,
):
    vehicle = _Vehicle()
    rcs = Hyper6Rcs()
    rcs.define(vehicle)
    _plant_externals(
        vehicle.store,
        ppcx=ppcx,
        qqcx=qqcx,
        rrcx=rrcx,
        phibdcx=phibdcx,
        thtbdcx=thtbdcx,
        psibdcx=psibdcx,
        psivdcx=psivdcx,
        utbc=utbc,
        ibbb=ibbb,
        fspcb=fspcb,
        aycomx=aycomx,
        azcomx=azcomx,
        vmass=vmass,
        minit=minit,
        beco_flag=beco_flag,
    )
    store = vehicle.store
    store.set("mrcs_moment", mrcs_moment)
    store.set("mrcs_force", mrcs_force)
    store.set("dead_zone", dead_zone)
    store.set("hysteresis", hysteresis)
    store.set("rcs_tau", rcs_tau)
    store.set("roll_mom_max", roll_mom_max)
    store.set("pitch_mom_max", pitch_mom_max)
    store.set("yaw_mom_max", yaw_mom_max)
    store.set("rcs_zeta", rcs_zeta)
    store.set("rcs_freq", rcs_freq)
    store.set("side_force_max", side_force_max)
    store.set("phibdcomx", phibdcomx)
    store.set("thtbdcomx", thtbdcomx)
    store.set("psibdcomx", psibdcomx)
    for name, value in states.items():
        store.set(name, value)
    rcs.initialize(vehicle, _ctx())
    return vehicle, rcs


def test_name_is_rcs():
    assert Hyper6Rcs().name == "rcs"


def test_hyper6_has_rcs_module_after_actuator():
    # Cape MODULES: actuator → rcs → forces.
    vehicle = Hyper6("Hypersonic", None, None)
    names = [m.name for m in vehicle.modules]
    assert "rcs" in names
    assert names.index("actuator") < names.index("rcs") < names.index("forces")
    assert type(vehicle.modules[names.index("rcs")]).__name__ == "Hyper6Rcs"


def test_define_registers_cpp_fields_not_externals():
    vehicle = _Vehicle()
    Hyper6Rcs().define(vehicle)
    store = vehicle.store
    for name in DEFINED:
        assert name in store.names(), name
        field = store.field(name)
        assert field.module == "rcs"
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        if name in INT_FIELDS:
            assert field.type == "int"
            if name == "rcs_minit_flag":
                assert store.get(name) == 1
            else:
                assert store.get(name) == 0
        elif name in VEC_FIELDS:
            assert field.type == "vec"
            assert np.array_equal(store.get(name), np.zeros(3))
        else:
            assert field.type == "real"
            assert store.get(name) == 0.0
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_rcs_prop_limits():
    assert _approx(rcs_prop(50.0, 100.0), 50.0)
    assert _approx(rcs_prop(150.0, 100.0), 100.0)
    assert _approx(rcs_prop(-150.0, 100.0), -100.0)


def test_rcs_schmitt_cadac_sign_zero_is_plus_one():
    assert rcs_schmitt(-1.0, 0.0, 0.0, 0.0) == 1


def test_mrcs_moment_11_prop_writes_fmrcs():
    # C++ rcs_type==1 / rcs_mode==1: nonzero FMRCS from prop Euler errors.
    phibdcomx = 5.0
    thtbdcomx = 8.0
    psibdcomx = -6.0
    vehicle, rcs = _ready(
        mrcs_moment=11,
        mrcs_force=0,
        phibdcomx=phibdcomx,
        thtbdcomx=thtbdcomx,
        psibdcomx=psibdcomx,
        roll_mom_max=ROLL_MOM_MAX,
        pitch_mom_max=200.0,
        yaw_mom_max=150.0,
        rcs_zeta=0.7,
        rcs_freq=10.0,
    )
    rcs.execute(vehicle, _ctx())
    store = vehicle.store
    rgain_roll = 2.0 * 0.7 * 10.0 * IBBB_DIAG[0]
    rgain_pitch = 2.0 * 0.7 * 10.0 * IBBB_DIAG[1]
    rgain_yaw = 2.0 * 0.7 * 10.0 * IBBB_DIAG[2]
    pgain = 10.0 / (2.0 * 0.7)
    e_roll = rgain_roll * (pgain * phibdcomx - 0.0)
    e_pitch = rgain_pitch * (pgain * thtbdcomx - 0.0)
    e_yaw = rgain_yaw * (pgain * psibdcomx - 0.0)
    want = np.array(
        [
            rcs_prop(e_roll, ROLL_MOM_MAX),
            rcs_prop(e_pitch, 200.0),
            rcs_prop(e_yaw, 150.0),
        ]
    )
    fmrcs = store.get("FMRCS")
    assert np.all(np.isfinite(fmrcs))
    np.testing.assert_allclose(fmrcs, want, rtol=RTOL, atol=ATOL)
    assert not np.allclose(fmrcs, 0.0)


def test_mrcs_moment_21_schmitt_writes_fmrcs():
    vehicle, rcs = _ready(
        mrcs_moment=21,
        thtbdcomx=THTBDCOMX,
        psibdcomx=PSIBCOMX,
        roll_save=0.0,
        pitch_save=THTBDCOMX,
        yaw_save=PSIBCOMX,
    )
    rcs.execute(vehicle, _ctx())
    fmrcs = vehicle.store.get("FMRCS")
    assert np.all(np.isfinite(fmrcs))
    assert _approx(fmrcs[0], 0.0)
    assert _approx(fmrcs[1], PITCH_MOM_MAX)
    assert _approx(fmrcs[2], YAW_MOM_MAX)
    assert fmrcs[1] != 0.0
    assert fmrcs[2] != 0.0


def test_mrcs_moment_12_prop_utbc_writes_fmrcs():
    utbc = (0.0, 0.1, -0.2)
    qqcx = 1.0
    rrcx = -0.5
    vehicle, rcs = _ready(
        mrcs_moment=12,
        utbc=utbc,
        qqcx=qqcx,
        rrcx=rrcx,
        pitch_mom_max=200.0,
        yaw_mom_max=150.0,
        rcs_zeta=0.7,
        rcs_freq=10.0,
    )
    rcs.execute(vehicle, _ctx())
    store = vehicle.store
    rgain_pitch = 2.0 * 0.7 * 10.0 * IBBB_DIAG[1]
    rgain_yaw = 2.0 * 0.7 * 10.0 * IBBB_DIAG[2]
    pgain = 10.0 / (2.0 * 0.7)
    e_pitch = rgain_pitch * (pgain * (-utbc[2]) * DEG - qqcx)
    e_yaw = rgain_yaw * (pgain * (utbc[1]) * DEG - rrcx)
    assert _approx(store.get("e_pitch"), e_pitch)
    assert _approx(store.get("e_yaw"), e_yaw)
    fmrcs = store.get("FMRCS")
    assert _approx(fmrcs[1], rcs_prop(e_pitch, 200.0))
    assert _approx(fmrcs[2], rcs_prop(e_yaw, 150.0))


def test_mrcs_moment_22_schmitt_utbc_writes_fmrcs():
    # C++ rcs_type==2 / rcs_mode==2 (TV/intercept): e_pitch/yaw from UTBC*DEG + rate.
    # e_pitch = -rcs_tau*qqcx - UTBC[2]*DEG; e_yaw = -rcs_tau*rrcx + UTBC[1]*DEG.
    utbc = (0.0, 0.1, -0.2)
    qqcx = 0.0
    rrcx = 0.0
    e_pitch = -RCS_TAU * qqcx - utbc[2] * DEG
    e_yaw = -RCS_TAU * rrcx + utbc[1] * DEG
    vehicle, rcs = _ready(
        mrcs_moment=22,
        qqcx=qqcx,
        rrcx=rrcx,
        utbc=utbc,
        thtbdcomx=THTBDCOMX,
        psibdcomx=PSIBCOMX,
        pitch_save=e_pitch,
        yaw_save=e_yaw,
    )
    rcs.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("e_pitch"), e_pitch)
    assert _approx(store.get("e_yaw"), e_yaw)
    assert store.get("e_pitch") != pytest.approx(THTBDCOMX, rel=RTOL, abs=ATOL)
    fmrcs = store.get("FMRCS")
    assert np.all(np.isfinite(fmrcs))
    assert _approx(fmrcs[1], PITCH_MOM_MAX * store.get("o_pitch"))
    assert _approx(fmrcs[2], YAW_MOM_MAX * store.get("o_yaw"))
    assert store.get("o_pitch") == 1
    assert store.get("o_yaw") == 1
    assert not np.allclose(fmrcs[1:], 0.0)


def test_mrcs_force_1_prop_uses_vmass_not_acc_gain():
    # C++ HYPER6: e_right = vmass*aycomx*AGRAV (open-loop), not rocket6 feedback.
    aycomx = 2.0
    azcomx = 1.5
    vehicle, rcs = _ready(
        mrcs_moment=0,
        mrcs_force=1,
        aycomx=aycomx,
        azcomx=azcomx,
        vmass=VMASS,
        fspcb=(0.0, 99.0, -99.0),
    )
    rcs.execute(vehicle, _ctx())
    store = vehicle.store
    e_right = VMASS * aycomx * AGRAV
    e_down = VMASS * azcomx * AGRAV
    want = np.array(
        [0.0, rcs_prop(e_right, SIDE_FORCE_MAX), rcs_prop(e_down, SIDE_FORCE_MAX)]
    )
    farcs = store.get("FARCS")
    np.testing.assert_allclose(farcs, want, rtol=RTOL, atol=ATOL)
    assert not np.allclose(farcs[1:], 0.0)
    assert _approx(store.get("e_right"), e_right)
    assert _approx(store.get("e_down"), e_down)
    np.testing.assert_allclose(store.get("FMRCS"), np.zeros(3), rtol=RTOL, atol=ATOL)


def test_minit_holds_psibdcomx_from_psivdcx():
    # C++: minit && rcs_mode==1 && (rcs_minit_flag||beco_flag) → psibdcomx=psivdcx.
    vehicle, rcs = _ready(
        mrcs_moment=21,
        minit=1,
        psivdcx=42.0,
        psibdcomx=PSIBCOMX,
        rcs_minit_flag=1,
        pitch_save=THTBDCOMX,
        yaw_save=42.0,
    )
    rcs.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("psibdcomx"), 42.0)
    assert store.get("rcs_minit_flag") == 0
    assert _approx(store.get("e_yaw"), 42.0)


def test_mrcs_moment_0_and_force_0_zeros():
    vehicle, rcs = _ready(mrcs_moment=0, mrcs_force=0, pitch_save=THTBDCOMX)
    rcs.execute(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("FMRCS"), np.zeros(3))
    np.testing.assert_array_equal(vehicle.store.get("FARCS"), np.zeros(3))

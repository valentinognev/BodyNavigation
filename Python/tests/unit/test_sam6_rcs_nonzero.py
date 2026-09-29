"""SAM6 RCS nonzero mrcs_moment / mrcs_force — Task 34."""

import numpy as np
import pytest

from cadac.constants import AGRAV, DEG
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.sam6.rcs import Sam6Rcs, rcs_prop, rcs_schmitt

RTOL = 1e-12
ATOL = 1e-14

DEAD_ZONE = 0.4
HYSTERESIS = 0.1
RCS_TAU = 1.0
ROLL_MOM_MAX = 100.0
PITCH_MOM_MAX = 200.0
YAW_MOM_MAX = 150.0
RCS_ZETA = 0.7
RCS_FREQ = 10.0
AI11 = 2.0
AI33 = 40.0
RATE_GAIN_RCS = 3.0
ACC_GAIN = 5.0
RCS_THRUST = 50.0
RCS_ARM = 4.0
XCG = 2.5
RCS_ISP = 200.0


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(int_step=0.001):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
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
    wbecb=(0.0, 0.0, 0.0),
    phiblcx=0.0,
    thtblcx=0.0,
    psibdcx=0.0,
    alphacx=0.0,
    betacx=0.0,
    fspcb=(0.0, 0.0, 0.0),
    ancomx=0.0,
    alcomx=0.0,
    utbc=(0.0, 0.0, 0.0),
    alphacomx=0.0,
    betacomx=0.0,
    ai11=AI11,
    ai33=AI33,
    xcg=XCG,
    pdynmc=0.0,
):
    store.define(Field("WBECB", wbecb, "vec", "out", "ins"))
    store.define(Field("phiblcx", phiblcx, "real", "out", "ins"))
    store.define(Field("thtblcx", thtblcx, "real", "out", "ins"))
    store.define(Field("psibdcx", psibdcx, "real", "out", "ins"))
    store.define(Field("alphacx", alphacx, "real", "out", "ins"))
    store.define(Field("betacx", betacx, "real", "out", "ins"))
    store.define(Field("FSPCB", fspcb, "vec", "out", "ins"))
    store.define(Field("ancomx", ancomx, "real", "out", "guidance"))
    store.define(Field("alcomx", alcomx, "real", "out", "guidance"))
    store.define(Field("UTBC", utbc, "vec", "out", "guidance"))
    store.define(Field("alphacomx", alphacomx, "real", "out", "guidance"))
    store.define(Field("betacomx", betacomx, "real", "out", "guidance"))
    store.define(Field("ai11", ai11, "real", "out", "propulsion"))
    store.define(Field("ai33", ai33, "real", "out", "propulsion"))
    store.define(Field("xcg", xcg, "real", "diag", "propulsion"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))


def _ready(
    *,
    mrcs_moment=0,
    mrcs_force=0,
    **kwargs,
):
    vehicle = _Vehicle()
    rcs = Sam6Rcs()
    rcs.define(vehicle)
    store = vehicle.store
    plant_keys = {
        "wbecb",
        "phiblcx",
        "thtblcx",
        "psibdcx",
        "alphacx",
        "betacx",
        "fspcb",
        "ancomx",
        "alcomx",
        "utbc",
        "alphacomx",
        "betacomx",
        "ai11",
        "ai33",
        "xcg",
        "pdynmc",
    }
    plant = {k: kwargs.pop(k) for k in list(kwargs) if k in plant_keys}
    _plant_externals(store, **plant)
    store.set("mrcs_moment", mrcs_moment)
    store.set("mrcs_force", mrcs_force)
    store.set("dead_zone", kwargs.pop("dead_zone", DEAD_ZONE))
    store.set("hysteresis", kwargs.pop("hysteresis", HYSTERESIS))
    store.set("rcs_tau", kwargs.pop("rcs_tau", RCS_TAU))
    store.set("roll_mom_max", kwargs.pop("roll_mom_max", ROLL_MOM_MAX))
    store.set("pitch_mom_max", kwargs.pop("pitch_mom_max", PITCH_MOM_MAX))
    store.set("yaw_mom_max", kwargs.pop("yaw_mom_max", YAW_MOM_MAX))
    store.set("rcs_zeta", kwargs.pop("rcs_zeta", RCS_ZETA))
    store.set("rcs_freq", kwargs.pop("rcs_freq", RCS_FREQ))
    store.set("rcs_arm", kwargs.pop("rcs_arm", RCS_ARM))
    store.set("rate_gain_rcs", kwargs.pop("rate_gain_rcs", RATE_GAIN_RCS))
    store.set("acc_gain", kwargs.pop("acc_gain", ACC_GAIN))
    store.set("rcs_thrust", kwargs.pop("rcs_thrust", RCS_THRUST))
    store.set("rcs_isp", kwargs.pop("rcs_isp", RCS_ISP))
    for name, value in kwargs.items():
        store.set(name, value)
    rcs.initialize(vehicle, _ctx())
    return vehicle, rcs


def test_rcs_prop_limits():
    assert _approx(rcs_prop(50.0, 100.0), 50.0)
    assert _approx(rcs_prop(150.0, 100.0), 100.0)
    assert _approx(rcs_prop(-150.0, 100.0), -100.0)


def test_rcs_schmitt_cadac_sign_zero_is_plus_one():
    assert rcs_schmitt(-1.0, 0.0, 0.0, 0.0) == 1


def test_sam6_rcs_nonzero_prop_euler_writes_fmrcs():
    """mrcs_moment=11 proportional geodetic Euler → finite nonzero FMRCS."""
    phibdcomx = 5.0
    thtbdcomx = 8.0
    psibdcomx = -6.0
    phiblcx = 0.0
    thtblcx = 0.0
    psibdcx = 0.0
    wbecb = (0.0, 0.0, 0.0)
    vehicle, rcs = _ready(
        mrcs_moment=11,
        mrcs_force=0,
        phibdcomx=phibdcomx,
        thtbdcomx=thtbdcomx,
        psibdcomx=psibdcomx,
        phiblcx=phiblcx,
        thtblcx=thtblcx,
        psibdcx=psibdcx,
        wbecb=wbecb,
    )
    rcs.execute(vehicle, _ctx())
    store = vehicle.store
    rgain_roll = 2.0 * RCS_ZETA * RCS_FREQ * AI11
    rgain_pitch = 2.0 * RCS_ZETA * RCS_FREQ * AI33
    rgain_yaw = rgain_pitch
    pgain = RCS_FREQ / (2.0 * RCS_ZETA)
    e_roll = rgain_roll * (pgain * (phibdcomx - phiblcx) - wbecb[0])
    e_pitch = rgain_pitch * (pgain * (thtbdcomx - thtblcx) - wbecb[1])
    e_yaw = rgain_yaw * (pgain * (psibdcomx - psibdcx) - wbecb[2])
    want = np.array(
        [
            rcs_prop(e_roll, ROLL_MOM_MAX),
            rcs_prop(e_pitch, PITCH_MOM_MAX),
            rcs_prop(e_yaw, YAW_MOM_MAX),
        ]
    )
    fmrcs = store.get("FMRCS")
    assert np.all(np.isfinite(fmrcs))
    np.testing.assert_allclose(fmrcs, want, rtol=RTOL, atol=ATOL)
    assert not np.allclose(fmrcs, 0.0)
    np.testing.assert_allclose(store.get("FARCS"), np.zeros(3), rtol=RTOL, atol=ATOL)


def test_sam6_rcs_nonzero_schmitt_euler_writes_fmrcs():
    """mrcs_moment=21 on-off Schmitt Euler → FMRCS = o * mom_max."""
    thtbdcomx = 80.0
    psibdcomx = -83.0
    vehicle, rcs = _ready(
        mrcs_moment=21,
        mrcs_force=0,
        phibdcomx=0.0,
        thtbdcomx=thtbdcomx,
        psibdcomx=psibdcomx,
        roll_save=0.0,
        pitch_save=thtbdcomx,
        yaw_save=psibdcomx,
    )
    rcs.execute(vehicle, _ctx())
    store = vehicle.store
    fmrcs = store.get("FMRCS")
    assert np.all(np.isfinite(fmrcs))
    assert _approx(fmrcs[0], 0.0)
    assert _approx(fmrcs[1], PITCH_MOM_MAX)
    assert _approx(fmrcs[2], -YAW_MOM_MAX)
    assert store.get("o_pitch") == 1
    assert store.get("o_yaw") == -1


def test_sam6_rcs_nonzero_prop_force_writes_farcs():
    """mrcs_force=1 proportional side thrusters → FARCS nonzero."""
    alcomx = 2.0
    ancomx = 1.5
    fspcb = (0.0, 1.0, -2.0)
    vehicle, rcs = _ready(
        mrcs_moment=0,
        mrcs_force=1,
        alcomx=alcomx,
        ancomx=ancomx,
        fspcb=fspcb,
    )
    rcs.execute(vehicle, _ctx())
    store = vehicle.store
    e_right = ACC_GAIN * (alcomx * AGRAV - fspcb[1])
    e_down = -ACC_GAIN * (ancomx * AGRAV + fspcb[2])
    want = np.array([0.0, rcs_prop(e_right, RCS_THRUST), rcs_prop(e_down, RCS_THRUST)])
    farcs = store.get("FARCS")
    np.testing.assert_allclose(farcs, want, rtol=RTOL, atol=ATOL)
    assert not np.allclose(farcs[1:], 0.0)
    np.testing.assert_allclose(store.get("FMRCS"), np.zeros(3), rtol=RTOL, atol=ATOL)


def test_sam6_rcs_nonzero_schmitt_force_parasitic_fmrcs():
    """mrcs_force=2 writes FARCS and parasitic FMRCS pitch/yaw."""
    alcomx = 3.0
    ancomx = 2.0
    fspcb = (0.0, 0.0, 0.0)
    e_right = ACC_GAIN * (alcomx * AGRAV - 0.0)
    e_down = -ACC_GAIN * (ancomx * AGRAV + 0.0)
    vehicle, rcs = _ready(
        mrcs_moment=0,
        mrcs_force=2,
        alcomx=alcomx,
        ancomx=ancomx,
        fspcb=fspcb,
        right_save=e_right,
        down_save=e_down,
        o_right=0,
        o_down=0,
    )
    rcs.execute(vehicle, _ctx(int_step=0.01))
    store = vehicle.store
    assert store.get("o_right") == 1
    assert store.get("o_down") == -1
    farcs = store.get("FARCS")
    assert _approx(farcs[1], RCS_THRUST)
    assert _approx(farcs[2], -RCS_THRUST)
    dx = RCS_ARM - XCG
    fmrcs = store.get("FMRCS")
    assert _approx(fmrcs[0], 0.0)
    assert _approx(fmrcs[1], farcs[2] * dx)
    assert _approx(fmrcs[2], -farcs[1] * dx)
    assert store.get("rcs_time") == pytest.approx(0.02, rel=RTOL, abs=ATOL)
    want_fmass = RCS_THRUST * 0.02 / (RCS_ISP * AGRAV)
    assert _approx(store.get("rcs_fmass"), want_fmass)


def test_sam6_rcs_nonzero_schmitt_incidence_mode_23():
    alphacomx = 10.0
    betacomx = 4.0
    e_pitch = alphacomx
    e_yaw = -betacomx
    vehicle, rcs = _ready(
        mrcs_moment=23,
        alphacomx=alphacomx,
        betacomx=betacomx,
        pitch_save=e_pitch,
        yaw_save=e_yaw,
    )
    rcs.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("e_pitch"), e_pitch)
    assert _approx(store.get("e_yaw"), e_yaw)
    fmrcs = store.get("FMRCS")
    assert _approx(fmrcs[1], PITCH_MOM_MAX)
    assert _approx(fmrcs[2], -YAW_MOM_MAX)


def test_sam6_rcs_nonzero_prop_utbc_mode_12():
    utbc = (0.0, 0.1, -0.2)
    vehicle, rcs = _ready(
        mrcs_moment=12,
        utbc=utbc,
        wbecb=(0.0, 0.0, 0.0),
        phibdcomx=0.0,
        phiblcx=0.0,
    )
    rcs.execute(vehicle, _ctx())
    store = vehicle.store
    rgain_pitch = 2.0 * RCS_ZETA * RCS_FREQ * AI33
    pgain = RCS_FREQ / (2.0 * RCS_ZETA)
    e_pitch = rgain_pitch * (pgain * (-utbc[2]) * DEG - 0.0)
    e_yaw = rgain_pitch * (pgain * (utbc[1]) * DEG - 0.0)
    assert _approx(store.get("e_pitch"), e_pitch)
    assert _approx(store.get("e_yaw"), e_yaw)
    fmrcs = store.get("FMRCS")
    assert _approx(fmrcs[1], rcs_prop(e_pitch, PITCH_MOM_MAX))
    assert _approx(fmrcs[2], rcs_prop(e_yaw, YAW_MOM_MAX))


def test_sam6_rcs_nonzero_rate_damping_modes():
    wbecb = (0.0, 5.0, -3.0)
    vehicle, rcs = _ready(
        mrcs_moment=14,
        wbecb=wbecb,
        phibdcomx=0.0,
        phiblcx=0.0,
    )
    rcs.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("e_pitch"), RATE_GAIN_RCS * (-wbecb[1]))
    assert _approx(store.get("e_yaw"), RATE_GAIN_RCS * (-wbecb[2]))
    vehicle2, rcs2 = _ready(
        mrcs_moment=24,
        wbecb=wbecb,
        rcs_tau=RCS_TAU,
        pitch_save=-RCS_TAU * wbecb[1],
        yaw_save=-RCS_TAU * wbecb[2],
    )
    rcs2.execute(vehicle2, _ctx())
    assert _approx(vehicle2.store.get("e_pitch"), -RCS_TAU * wbecb[1])
    assert _approx(vehicle2.store.get("e_yaw"), -RCS_TAU * wbecb[2])

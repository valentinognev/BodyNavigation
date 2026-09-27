from math import atan2, cos, sin, sqrt
from types import SimpleNamespace

import pytest

from cadac.constants import AGRAV, DEG, RAD
from cadac.env.us76 import atmosphere76
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.sraam6.control import SMALL, Sraam6Control

RTOL = 1e-12
ATOL = 1e-14

PHICOMX = 0.0
WRCL = 20.0
ZRCL = 0.9
DLP = -2.0
DLD = 1.0
PHIBLX = 10.0
PP = 0.0
ZETLAGR = 0.6
DNA = 50.0
DND = 10.0
DMA = -80.0
DMQ = -3.0
DMD = -40.0
DVBE = 250.0
QQ = 0.2
RR = -0.1
ALIMIT = 50.0
ANCOMX = 1.0
ALCOMX = 0.0
HBE = 5000.0
DT = 0.001
GAINP = 0.0
FACTWACL = 0.0
FACTZACL = 0.0
FSPB = (0.0, 0.0, 0.0)
SENTINEL_DPCX = 9.0
SENTINEL_DQCX = 7.0

_RHO, _PRESS, _TEMPK = atmosphere76(HBE)
PDYNMC = 0.5 * _RHO * DVBE**2


def _ctx(int_step=DT):
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


def _externals(store):
    store.define(Field("phiblx", PHIBLX, "real", "diag", "kinematics"))
    store.define(Field("pp", PP, "real", "state", "euler"))
    store.define(Field("dlp", DLP, "real", "out", "aerodynamics"))
    store.define(Field("dld", DLD, "real", "out", "aerodynamics"))
    store.define(Field("dna", DNA, "real", "out", "aerodynamics"))
    store.define(Field("dnd", DND, "real", "out", "aerodynamics"))
    store.define(Field("dma", DMA, "real", "out", "aerodynamics"))
    store.define(Field("dmq", DMQ, "real", "out", "aerodynamics"))
    store.define(Field("dmd", DMD, "real", "out", "aerodynamics"))
    store.define(Field("dvbe", DVBE, "real", "out", "newton"))
    store.define(Field("qq", QQ, "real", "state", "euler"))
    store.define(Field("rr", RR, "real", "state", "euler"))
    store.define(Field("pdynmc", PDYNMC, "real", "out", "environment"))
    store.define(Field("ancomx", ANCOMX, "real", "out", "guidance"))
    store.define(Field("alcomx", ALCOMX, "real", "out", "guidance"))
    store.define(Field("FSPB", FSPB, "vec", "out", "newton"))


def _ready(maut):
    vehicle = SimpleNamespace(store=StateStore())
    ctrl = Sraam6Control()
    ctrl.define(vehicle)
    _externals(vehicle.store)
    store = vehicle.store
    store.set("maut", maut)
    store.set("phicomx", PHICOMX)
    store.set("wrcl", WRCL)
    store.set("zrcl", ZRCL)
    store.set("zetlagr", ZETLAGR)
    store.set("alimit", ALIMIT)
    store.set("gainp", GAINP)
    store.set("factwacl", FACTWACL)
    store.set("factzacl", FACTZACL)
    ctrl.initialize(vehicle, _ctx())
    return vehicle, ctrl


def _cpp_control_roll():
    gkp = (2.0 * ZRCL * WRCL + DLP) / DLD
    gkphi = WRCL * WRCL / DLD
    ephi = gkphi * (PHICOMX - PHIBLX) * RAD
    dpc = ephi - gkp * PP
    return dpc * DEG


def _cpp_control_rate():
    zrate = DNA / DVBE - DMA * DND / (DVBE * DMD)
    aa = DNA / DVBE - DMQ
    bb = -DMA - DMQ * DNA / DVBE
    dum1 = aa - 2.0 * ZETLAGR * ZETLAGR * zrate
    dum2 = aa * aa - 4.0 * ZETLAGR * ZETLAGR * bb
    radix = dum1 * dum1 - dum2
    if radix < 0.0:
        radix = 0.0
    dmd = DMD
    if abs(dmd) < SMALL:
        dmd = SMALL * (1 if dmd >= 0.0 else -1)
    grate = (-dum1 + sqrt(radix)) / (-dmd)
    return DEG * grate * QQ, DEG * grate * RR


def _cpp_control_accel(store, dt):
    alcomx = ALCOMX
    ancomx = ANCOMX
    aa = sqrt(alcomx * alcomx + ancomx * ancomx)
    if aa > ALIMIT:
        aa = ALIMIT
    if abs(ancomx) < SMALL and abs(alcomx) < SMALL:
        phi = 0.0
    else:
        phi = atan2(ancomx, alcomx)
    alcomx = aa * cos(phi)
    ancomx = aa * sin(phi)
    wacl = (0.013 * sqrt(PDYNMC) + 7.1) * (FACTWACL + 1)
    zacl = (0.559e-3 * sqrt(PDYNMC) + 0.232) * (FACTZACL + 1)
    pacl = 14
    gainfb3 = wacl * wacl * pacl / (DNA * DMD)
    gainfb2 = (2.0 * zacl * wacl + pacl + DMQ - DNA / DVBE) / DMD
    gainfb1 = (
        wacl * wacl
        + 2.0 * zacl * wacl * pacl
        + DMA
        + DMQ * DNA / DVBE
        - gainfb2 * DMD * DNA / DVBE
    ) / (DNA * DMD) - GAINP
    fspb = store.get("FSPB")
    fspb3 = fspb[2]
    zzd_new = AGRAV * ancomx + fspb3
    zz = integrate(zzd_new, store.get("zzd"), store.get("zz"), dt)
    dqc = -gainfb1 * (-fspb3) - gainfb2 * QQ + gainfb3 * zz
    return dqc * DEG


def test_maut_0_leaves_dqcx_unchanged():
    vehicle, ctrl = _ready(0)
    store = vehicle.store
    store.set("dqcx", SENTINEL_DQCX)
    store.set("dpcx", SENTINEL_DPCX)
    assert ctrl.execute(vehicle, _ctx()) is None
    assert store.get("dqcx") == SENTINEL_DQCX
    assert store.get("dpcx") == SENTINEL_DPCX


def test_maut_1_writes_dpcx():
    vehicle, ctrl = _ready(1)
    store = vehicle.store
    store.set("dqcx", SENTINEL_DQCX)
    want_dpcx = _cpp_control_roll()
    assert want_dpcx != 0.0
    assert ctrl.execute(vehicle, _ctx()) is None
    assert _approx(store.get("dpcx"), want_dpcx)
    assert store.get("dpcx") != SENTINEL_DPCX
    assert store.get("dqcx") == SENTINEL_DQCX


def test_maut_2_writes_dqcx():
    vehicle, ctrl = _ready(2)
    store = vehicle.store
    store.set("dqcx", SENTINEL_DQCX)
    want_dpcx = _cpp_control_roll()
    want_dqcx, want_drcx = _cpp_control_rate()
    want_accel = _cpp_control_accel(store, DT)
    assert want_dqcx != 0.0
    assert want_dqcx != pytest.approx(want_accel, rel=RTOL, abs=ATOL)
    assert ctrl.execute(vehicle, _ctx()) is None
    assert _approx(store.get("dpcx"), want_dpcx)
    assert _approx(store.get("dqcx"), want_dqcx)
    assert _approx(store.get("drcx"), want_drcx)
    assert store.get("dqcx") != SENTINEL_DQCX


def test_maut_3_writes_dqcx():
    vehicle, ctrl = _ready(3)
    store = vehicle.store
    store.set("dqcx", SENTINEL_DQCX)
    want_dpcx = _cpp_control_roll()
    want_dqcx = _cpp_control_accel(store, DT)
    want_rate, _want_drcx = _cpp_control_rate()
    assert want_dqcx != 0.0
    assert want_dqcx != pytest.approx(want_rate, rel=RTOL, abs=ATOL)
    assert ctrl.execute(vehicle, _ctx()) is None
    assert _approx(store.get("dpcx"), want_dpcx)
    assert _approx(store.get("dqcx"), want_dqcx)
    assert store.get("dqcx") != SENTINEL_DQCX


@pytest.mark.parametrize("maut", (4, -1))
def test_maut_unused_raises(maut):
    vehicle, ctrl = _ready(maut)
    store = vehicle.store
    store.set("dqcx", SENTINEL_DQCX)
    store.set("dpcx", SENTINEL_DPCX)
    with pytest.raises(ValueError, match="unknown maut"):
        ctrl.execute(vehicle, _ctx())
    assert store.get("dqcx") == SENTINEL_DQCX
    assert store.get("dpcx") == SENTINEL_DPCX

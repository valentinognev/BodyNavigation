from math import atan2, cos, sin, sqrt
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import AGRAV, DEG
from cadac.env.us76 import atmosphere76
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.sraam6.control import SMALL, Sraam6Control

RTOL = 1e-12
ATOL = 1e-14

ALIMIT = 50.0
ANCOMX = 1.0
ALCOMX = 0.0
HBE = 5000.0
DVBE = 250.0
DT = 0.001
DNA = 50.0
DMA = -80.0
DMQ = -3.0
DMD = -40.0
GAINP = 0.0
FACTWACL = 0.0
FACTZACL = 0.0
QQ = 0.0
RR = 0.0
FSPB = (0.0, 0.0, 0.0)
CLIP_ANCOMX = 60.0
CLIP_ALCOMX = 0.0

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


def _externals(
    store,
    *,
    pdynmc=PDYNMC,
    ancomx=ANCOMX,
    alcomx=ALCOMX,
    dna=DNA,
    dma=DMA,
    dmq=DMQ,
    dmd=DMD,
    dvbe=DVBE,
    qq=QQ,
    rr=RR,
    fspb=FSPB,
):
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("ancomx", ancomx, "real", "out", "guidance"))
    store.define(Field("alcomx", alcomx, "real", "out", "guidance"))
    store.define(Field("dna", dna, "real", "out", "aerodynamics"))
    store.define(Field("dma", dma, "real", "out", "aerodynamics"))
    store.define(Field("dmq", dmq, "real", "out", "aerodynamics"))
    store.define(Field("dmd", dmd, "real", "out", "aerodynamics"))
    store.define(Field("dvbe", dvbe, "real", "out", "newton"))
    store.define(Field("qq", qq, "real", "state", "euler"))
    store.define(Field("rr", rr, "real", "state", "euler"))
    store.define(Field("FSPB", fspb, "vec", "out", "newton"))


def _ready(**kw):
    vehicle = SimpleNamespace(store=StateStore())
    ctrl = Sraam6Control()
    ctrl.define(vehicle)
    _externals(vehicle.store, **kw)
    store = vehicle.store
    store.set("alimit", ALIMIT)
    store.set("gainp", GAINP)
    store.set("factwacl", FACTWACL)
    store.set("factzacl", FACTZACL)
    ctrl.initialize(vehicle, _ctx())
    return vehicle, ctrl


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _cpp_control_accel(store, dt):
    alimit = store.get("alimit")
    gainp = store.get("gainp")
    factwacl = store.get("factwacl")
    factzacl = store.get("factzacl")
    pdynmc = store.get("pdynmc")
    ancomx = store.get("ancomx")
    alcomx = store.get("alcomx")
    dna = store.get("dna")
    dma = store.get("dma")
    dmq = store.get("dmq")
    dmd = store.get("dmd")
    fspb = store.get("FSPB")
    dvbe = store.get("dvbe")
    qq = store.get("qq")
    rr = store.get("rr")
    yyd = store.get("yyd")
    yy = store.get("yy")
    zzd = store.get("zzd")
    zz = store.get("zz")

    aa = sqrt(alcomx * alcomx + ancomx * ancomx)
    if aa > alimit:
        aa = alimit
    if abs(ancomx) < SMALL and abs(alcomx) < SMALL:
        phi = 0.0
    else:
        phi = atan2(ancomx, alcomx)
    alcomx = aa * cos(phi)
    ancomx = aa * sin(phi)

    wacl = (0.013 * sqrt(pdynmc) + 7.1) * (factwacl + 1)
    zacl = (0.559e-3 * sqrt(pdynmc) + 0.232) * (factzacl + 1)
    pacl = 14

    gainfb3 = wacl * wacl * pacl / (dna * dmd)
    gainfb2 = (2.0 * zacl * wacl + pacl + dmq - dna / dvbe) / dmd
    gainfb1 = (
        wacl * wacl
        + 2.0 * zacl * wacl * pacl
        + dma
        + dmq * dna / dvbe
        - gainfb2 * dmd * dna / dvbe
    ) / (dna * dmd) - gainp
    gainfb = (gainfb1, gainfb2, gainfb3)

    fspb3 = fspb[2]
    zzd_new = AGRAV * ancomx + fspb3
    zz = integrate(zzd_new, zzd, zz, dt)
    zzd = zzd_new
    dqc = -gainfb1 * (-fspb3) - gainfb2 * qq + gainfb3 * zz
    dqcx = dqc * DEG

    fspb2 = fspb[1]
    yyd_new = AGRAV * alcomx - fspb2
    yy = integrate(yyd_new, yyd, yy, dt)
    yyd = yyd_new
    drc = -gainfb1 * fspb2 - gainfb2 * rr + gainfb3 * yy
    drcx = drc * DEG

    return {
        "ancomx": ancomx,
        "alcomx": alcomx,
        "wacl": wacl,
        "zacl": zacl,
        "pacl": pacl,
        "GAINFB": gainfb,
        "yy": yy,
        "yyd": yyd,
        "zz": zz,
        "zzd": zzd,
        "dqcx": dqcx,
        "drcx": drcx,
    }


def test_execute_remains_pass():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.set("dqcx", 7.0)
    store.set("drcx", 8.0)
    store.set("wacl", 1.5)
    store.set("zacl", 2.5)
    store.set("pacl", 3.5)
    store.set("yy", 4.0)
    store.set("zz", 5.0)
    assert ctrl.execute(vehicle, _ctx()) is None
    assert store.get("dqcx") == 7.0
    assert store.get("drcx") == 8.0
    assert store.get("wacl") == 1.5
    assert store.get("zacl") == 2.5
    assert store.get("pacl") == 3.5
    assert store.get("yy") == 4.0
    assert store.get("zz") == 5.0


def test_control_accel_1v1_one_step_matches_cpp():
    vehicle, ctrl = _ready()
    want = _cpp_control_accel(vehicle.store, DT)
    assert ctrl.control_accel(vehicle, _ctx()) is None
    store = vehicle.store
    assert np.isfinite(store.get("dqcx"))
    assert np.isfinite(want["dqcx"])
    assert _approx(store.get("dqcx"), want["dqcx"])
    assert _approx(store.get("drcx"), want["drcx"])
    assert _approx(store.get("wacl"), want["wacl"])
    assert _approx(store.get("zacl"), want["zacl"])
    assert store.get("pacl") == 14
    assert _approx(store.get("pacl"), want["pacl"])
    np.testing.assert_allclose(store.get("GAINFB"), want["GAINFB"], rtol=RTOL, atol=ATOL)
    assert _approx(store.get("yy"), want["yy"])
    assert _approx(store.get("yyd"), want["yyd"])
    assert _approx(store.get("zz"), want["zz"])
    assert _approx(store.get("zzd"), want["zzd"])
    assert abs(want["ancomx"]) <= ALIMIT
    assert abs(want["alcomx"]) <= ALIMIT
    assert store.get("ancomx") == ANCOMX
    assert store.get("alcomx") == ALCOMX
    assert store.get("dpcx") == 0.0
    assert store.get("pdynmc") == PDYNMC
    assert store.get("alimit") == ALIMIT
    assert store.get("dvbe") == DVBE


def test_control_accel_overwrites_poles_from_pdynmc():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.set("wacl", 99.0)
    store.set("zacl", 88.0)
    store.set("pacl", 77.0)
    want = _cpp_control_accel(store, DT)
    ctrl.control_accel(vehicle, _ctx())
    assert store.get("wacl") != 99.0
    assert store.get("zacl") != 88.0
    assert store.get("pacl") != 77.0
    assert _approx(store.get("wacl"), want["wacl"])
    assert _approx(store.get("zacl"), want["zacl"])
    assert store.get("pacl") == 14


def test_control_accel_limiter_clips_to_alimit():
    vehicle, ctrl = _ready(ancomx=CLIP_ANCOMX, alcomx=CLIP_ALCOMX)
    want = _cpp_control_accel(vehicle.store, DT)
    aa_raw = sqrt(CLIP_ANCOMX * CLIP_ANCOMX + CLIP_ALCOMX * CLIP_ALCOMX)
    assert aa_raw > ALIMIT
    assert abs(want["ancomx"]) <= ALIMIT
    assert abs(want["alcomx"]) <= ALIMIT
    ctrl.control_accel(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("dqcx"), want["dqcx"])
    assert _approx(store.get("drcx"), want["drcx"])
    assert abs(want["ancomx"]) == pytest.approx(ALIMIT, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == CLIP_ANCOMX
    assert store.get("alcomx") == CLIP_ALCOMX

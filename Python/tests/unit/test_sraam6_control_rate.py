from math import sqrt
from types import SimpleNamespace

import cadac.constants as cadac_constants
import pytest

from cadac.constants import DEG
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.sraam6.control import SMALL, Sraam6Control

RTOL = 1e-12
ATOL = 1e-14

ZETLAGR = 0.6
DNA = 50.0
DND = 10.0
DMA = -80.0
DMQ = -3.0
DMD = -40.0
DVBE = 250.0
QQ = 0.2
RR = -0.1
TINY_DMD = 1.0e-9


def _sign(variable):
    if variable < 0.0:
        return -1
    return 1


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.001,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _externals(
    store,
    *,
    dna=DNA,
    dnd=DND,
    dma=DMA,
    dmq=DMQ,
    dmd=DMD,
    dvbe=DVBE,
    qq=QQ,
    rr=RR,
):
    store.define(Field("dna", dna, "real", "out", "aerodynamics"))
    store.define(Field("dnd", dnd, "real", "out", "aerodynamics"))
    store.define(Field("dma", dma, "real", "out", "aerodynamics"))
    store.define(Field("dmq", dmq, "real", "out", "aerodynamics"))
    store.define(Field("dmd", dmd, "real", "out", "aerodynamics"))
    store.define(Field("dvbe", dvbe, "real", "out", "newton"))
    store.define(Field("qq", qq, "real", "state", "euler"))
    store.define(Field("rr", rr, "real", "state", "euler"))


def _ready(**kw):
    vehicle = SimpleNamespace(store=StateStore())
    ctrl = Sraam6Control()
    ctrl.define(vehicle)
    _externals(vehicle.store, **kw)
    vehicle.store.set("zetlagr", ZETLAGR)
    ctrl.initialize(vehicle, _ctx())
    return vehicle, ctrl


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _cpp_control_rate(store):
    zetlagr = store.get("zetlagr")
    qq = store.get("qq")
    rr = store.get("rr")
    dvbe = store.get("dvbe")
    dna = store.get("dna")
    dnd = store.get("dnd")
    dma = store.get("dma")
    dmq = store.get("dmq")
    dmd = store.get("dmd")
    zrate = dna / dvbe - dma * dnd / (dvbe * dmd)
    aa = dna / dvbe - dmq
    bb = -dma - dmq * dna / dvbe
    dum1 = aa - 2.0 * zetlagr * zetlagr * zrate
    dum2 = aa * aa - 4.0 * zetlagr * zetlagr * bb
    radix = dum1 * dum1 - dum2
    if radix < 0.0:
        radix = 0.0
    if abs(dmd) < SMALL:
        dmd = SMALL * _sign(dmd)
    grate = (-dum1 + sqrt(radix)) / (-dmd)
    dum3 = grate * dmd * zrate
    radix = bb + dum3
    if radix < 0.0:
        radix = 0.0
    wnlagr = sqrt(radix)
    dqcx = DEG * grate * qq
    drcx = DEG * grate * rr
    return zrate, grate, wnlagr, dqcx, drcx


def test_small_is_module_level_not_in_constants():
    assert SMALL == 1e-7
    assert not hasattr(cadac_constants, "SMALL")


def test_execute_remains_pass():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.set("dqcx", 7.0)
    store.set("drcx", 8.0)
    store.set("grate", 1.5)
    store.set("zrate", 2.5)
    store.set("wnlagr", 3.5)
    assert ctrl.execute(vehicle, _ctx()) is None
    assert store.get("dqcx") == 7.0
    assert store.get("drcx") == 8.0
    assert store.get("grate") == 1.5
    assert store.get("zrate") == 2.5
    assert store.get("wnlagr") == 3.5


def test_control_rate_frozen_matches_cpp_grate_dqcx_drcx():
    vehicle, ctrl = _ready()
    want_zrate, want_grate, want_wnlagr, want_dqcx, want_drcx = _cpp_control_rate(
        vehicle.store
    )
    assert ctrl.control_rate(vehicle) is None
    store = vehicle.store
    assert _approx(store.get("grate"), want_grate)
    assert _approx(store.get("dqcx"), want_dqcx)
    assert _approx(store.get("drcx"), want_drcx)
    assert _approx(store.get("zrate"), want_zrate)
    assert _approx(store.get("wnlagr"), want_wnlagr)
    assert store.get("dqcx") != 0.0
    assert store.get("drcx") != 0.0
    assert store.get("grate") != 0.0
    assert store.get("zetlagr") == ZETLAGR
    assert store.get("dvbe") == DVBE
    assert store.get("dmd") == DMD
    assert store.get("dpcx") == 0.0


def test_control_rate_clamps_tiny_dmd():
    vehicle, ctrl = _ready(dmd=TINY_DMD)
    want_zrate, want_grate, want_wnlagr, want_dqcx, want_drcx = _cpp_control_rate(
        vehicle.store
    )
    ctrl.control_rate(vehicle)
    store = vehicle.store
    assert abs(TINY_DMD) < SMALL
    assert _approx(store.get("grate"), want_grate)
    assert _approx(store.get("dqcx"), want_dqcx)
    assert _approx(store.get("drcx"), want_drcx)
    assert _approx(store.get("zrate"), want_zrate)
    assert _approx(store.get("wnlagr"), want_wnlagr)
    assert store.get("dmd") == TINY_DMD
    zrate = DNA / DVBE - DMA * DND / (DVBE * TINY_DMD)
    aa = DNA / DVBE - DMQ
    bb = -DMA - DMQ * DNA / DVBE
    dum1 = aa - 2.0 * ZETLAGR * ZETLAGR * zrate
    dum2 = aa * aa - 4.0 * ZETLAGR * ZETLAGR * bb
    radix = dum1 * dum1 - dum2
    if radix < 0.0:
        radix = 0.0
    unclamped = (-dum1 + sqrt(radix)) / (-TINY_DMD)
    assert store.get("grate") != pytest.approx(unclamped, rel=RTOL, abs=ATOL)


def test_control_rate_reads_qq_rr_not_ins():
    vehicle, ctrl = _ready(qq=QQ, rr=RR)
    vehicle.store.define(Field("WBECB", (0.0, 0.9, -0.8), "vec", "out", "ins"))
    vehicle.store.define(Field("qqx", 99.0, "real", "out", "euler"))
    vehicle.store.define(Field("rrx", -99.0, "real", "out", "euler"))
    _want_zrate, want_grate, _want_wnlagr, want_dqcx, want_drcx = _cpp_control_rate(
        vehicle.store
    )
    ctrl.control_rate(vehicle)
    store = vehicle.store
    assert _approx(store.get("grate"), want_grate)
    assert _approx(store.get("dqcx"), want_dqcx)
    assert _approx(store.get("drcx"), want_drcx)
    decoy_dqcx = DEG * want_grate * 0.9
    decoy_drcx = DEG * want_grate * -0.8
    assert store.get("dqcx") != pytest.approx(decoy_dqcx, rel=RTOL, abs=ATOL)
    assert store.get("drcx") != pytest.approx(decoy_drcx, rel=RTOL, abs=ATOL)

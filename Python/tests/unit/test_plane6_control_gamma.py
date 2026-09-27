from math import isfinite
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.falcon6.control import Plane6Control

RTOL = 1e-12
ATOL = 1e-14

# input_gamma.asc gamma poles; Step 1 command is 1 deg (not the 10 deg case command).
PGAM = 10.0
WGAM = 3.0
ZGAM = 0.5
THTVLCOMX = 1.0
THTBLX = 1.0
QQX = 4.0
THTVLX = 0.5
DLA = 8.0
DLDE = -12.0
DMA = -2.0
DMQ = -1.5
DMDE = -8.0
DVBE = 180.0


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
    thtblx=THTBLX,
    qqx=QQX,
    thtvlx=THTVLX,
    dvbe=DVBE,
    dla=DLA,
    dlde=DLDE,
    dma=DMA,
    dmq=DMQ,
    dmde=DMDE,
):
    store.define(Field("thtblx", thtblx, "real", "diag", "kinematics"))
    store.define(Field("qqx", qqx, "real", "out", "euler"))
    store.define(Field("thtvlx", thtvlx, "real", "out", "newton"))
    store.define(Field("dvbe", dvbe, "real", "out", "newton"))
    store.define(Field("dla", dla, "real", "out", "aerodynamics"))
    store.define(Field("dlde", dlde, "real", "out", "aerodynamics"))
    store.define(Field("dma", dma, "real", "out", "aerodynamics"))
    store.define(Field("dmq", dmq, "real", "out", "aerodynamics"))
    store.define(Field("dmde", dmde, "real", "out", "aerodynamics"))


def _ready(**kw):
    vehicle = SimpleNamespace(store=StateStore())
    ctrl = Plane6Control()
    ctrl.define(vehicle)
    _externals(vehicle.store, **kw)
    store = vehicle.store
    store.set("pgam", PGAM)
    store.set("wgam", WGAM)
    store.set("zgam", ZGAM)
    ctrl.initialize(vehicle, _ctx())
    return vehicle, ctrl


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _control_gamma(store, thtvlcomx):
    pgam = store.get("pgam")
    wgam = store.get("wgam")
    zgam = store.get("zgam")
    thtblx = store.get("thtblx")
    qqx = store.get("qqx")
    thtvlx = store.get("thtvlx")
    dvbe = store.get("dvbe")
    dla = store.get("dla")
    dlde = store.get("dlde")
    dma = store.get("dma")
    dmq = store.get("dmq")
    dmde = store.get("dmde")

    aa = np.array(
        [
            [dmq, dma, -dma],
            [1.0, 0.0, 0.0],
            [0.0, dla / dvbe, -dla / dvbe],
        ],
        dtype=float,
    )
    bb = np.array([dmde, 0.0, dlde / dvbe], dtype=float)

    am = 2.0 * zgam * wgam + pgam
    bm = wgam * wgam + 2.0 * zgam * wgam * pgam
    cm = wgam * wgam * pgam
    v11 = dmde
    v12 = 0.0
    v13 = dlde / dvbe
    v21 = dmde * dla / dvbe - dlde * dma / dvbe
    v22 = dmde
    v23 = -dmq * dlde / dvbe
    v31 = 0.0
    v32 = v21
    v33 = v21
    dp = np.array(
        [
            [v11, v12, v13],
            [v21, v22, v23],
            [v31, v32, v33],
        ],
        dtype=float,
    )
    dd = np.array(
        [am + dmq - dla / dvbe, bm + dma + dmq * dla / dvbe, cm],
        dtype=float,
    )
    gaingam = np.linalg.inv(dp) @ dd
    dum33 = aa - np.outer(bb, gaingam)
    dum3 = np.linalg.inv(dum33) @ bb
    hh = np.array([0.0, 0.0, 1.0], dtype=float)
    gainff = -1.0 / (hh @ dum3)

    thtc = gainff * thtvlcomx * RAD
    qqf = gaingam[0] * qqx * RAD
    thtblf = gaingam[1] * thtblx * RAD
    thtvlf = gaingam[2] * thtvlx * RAD
    delec = thtc - (qqf + thtblf + thtvlf)
    delecx = delec * DEG
    return delecx, gaingam, gainff


def test_control_gamma_thtvlcomx_one_deg_elevator_finite_matches_cadac():
    vehicle, ctrl = _ready()
    store = vehicle.store
    assert store.get("dvbe") != 0.0
    assert store.get("dla") != 0.0
    assert store.get("dmde") != 0.0
    assert "time" not in store.names()
    assert "pdynmc" not in store.names()
    want, want_gaingam, want_gainff = _control_gamma(store, THTVLCOMX)
    got = ctrl.control_gamma(vehicle, THTVLCOMX)
    assert isfinite(got)
    assert _approx(got, want)
    np.testing.assert_allclose(store.get("GAINGAM"), want_gaingam, rtol=RTOL, atol=ATOL)
    assert store.get("GAINGAM").shape == (3,)
    assert _approx(store.get("gainff"), want_gainff)
    assert want_gainff != 0.0
    assert np.any(want_gaingam != 0.0)
    assert want != 0.0
    assert store.get("delecx") == 0.0


def test_control_gamma_does_not_write_delecx():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.set("delecx", 7.0)
    ctrl.control_gamma(vehicle, THTVLCOMX)
    assert store.get("delecx") == 7.0


def test_execute_dispatches_maut_24_with_gamma_inputs():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.define(Field("phiblx", 0.0, "real", "diag", "kinematics"))
    store.define(Field("ppx", 3.0, "real", "out", "euler"))
    store.define(Field("dllp", -2.5, "real", "out", "aerodynamics"))
    store.define(Field("dllda", 18.0, "real", "out", "aerodynamics"))
    store.define(Field("dyb", -6.0, "real", "out", "aerodynamics"))
    store.define(Field("dydr", 4.0, "real", "out", "aerodynamics"))
    store.define(Field("dnb", 1.5, "real", "out", "aerodynamics"))
    store.define(Field("dnr", -0.8, "real", "out", "aerodynamics"))
    store.define(Field("dndr", -3.0, "real", "out", "aerodynamics"))
    store.define(Field("rrx", 2.0, "real", "out", "euler"))
    store.set("wrcl", 15.0)
    store.set("zrcl", 0.7)
    store.set("zetlagr", 0.7)
    store.set("dalimx", 1.0e6)
    store.set("delimx", 1.0e6)
    store.set("drlimx", 1.0e6)
    store.set("philimx", 70.0)
    store.set("maut", 24)
    store.set("thtvlcomx", THTVLCOMX)
    ctrl.execute(vehicle, _ctx())
    want, want_gaingam, want_gainff = _control_gamma(store, THTVLCOMX)
    assert isfinite(store.get("delecx"))
    assert _approx(store.get("delecx"), want)
    np.testing.assert_allclose(store.get("GAINGAM"), want_gaingam, rtol=RTOL, atol=ATOL)
    assert _approx(store.get("gainff"), want_gainff)
    assert store.get("delecx") != 0.0

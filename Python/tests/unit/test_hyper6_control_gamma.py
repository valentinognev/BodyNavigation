from math import isfinite
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.hyper6.control import Hyper6Control

RTOL = 1e-12
ATOL = 1e-14

# input_climb.asc gamma poles; Step 1 command is 0 deg (event at t>10 sets 10).
PGAM = 4.0
WGAM = 2.0
ZGAM = 0.7
THTVDCOMX = 0.0
THTVDCOMX_EVENT = 10.0
THTBDCX = 2.5
QQCX = 4.0
THTVDCX = 0.0
DLA = 8.0
DLDE = -12.0
DMA = -2.0
DMQ = -1.5
DMDE = -8.0
DVBEC = 1000.0
DVBE = 1000.0


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.01,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _externals(
    store,
    *,
    thtbdcx=THTBDCX,
    qqcx=QQCX,
    thtvdcx=THTVDCX,
    dvbec=DVBEC,
    dvbe=DVBE,
    dla=DLA,
    dlde=DLDE,
    dma=DMA,
    dmq=DMQ,
    dmde=DMDE,
):
    store.define(Field("thtbdcx", thtbdcx, "real", "out", "ins"))
    store.define(Field("qqcx", qqcx, "real", "out", "ins"))
    store.define(Field("thtvdcx", thtvdcx, "real", "out", "ins"))
    store.define(Field("dvbec", dvbec, "real", "out", "ins"))
    store.define(Field("dvbe", dvbe, "real", "out", "newton"))
    store.define(Field("dla", dla, "real", "out", "aerodynamics"))
    store.define(Field("dlde", dlde, "real", "out", "aerodynamics"))
    store.define(Field("dma", dma, "real", "out", "aerodynamics"))
    store.define(Field("dmq", dmq, "real", "out", "aerodynamics"))
    store.define(Field("dmde", dmde, "real", "out", "aerodynamics"))


def _ready(**kw):
    vehicle = SimpleNamespace(store=StateStore())
    ctrl = Hyper6Control()
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


def _control_gamma(store, thtvdcomx):
    pgam = store.get("pgam")
    wgam = store.get("wgam")
    zgam = store.get("zgam")
    thtbdcx = store.get("thtbdcx")
    qqcx = store.get("qqcx")
    thtvdcx = store.get("thtvdcx")
    dvbec = store.get("dvbec")
    dvbe = store.get("dvbe")
    dla = store.get("dla")
    dlde = store.get("dlde")
    dma = store.get("dma")
    dmq = store.get("dmq")
    dmde = store.get("dmde")

    if dvbec == 0:
        dvbec = dvbe

    aa = np.array(
        [
            [dmq, dma, -dma],
            [1.0, 0.0, 0.0],
            [0.0, dla / dvbec, -dla / dvbec],
        ],
        dtype=float,
    )
    bb = np.array([dmde, 0.0, dlde / dvbec], dtype=float)

    am = 2.0 * zgam * wgam + pgam
    bm = wgam * wgam + 2.0 * zgam * wgam * pgam
    cm = wgam * wgam * pgam
    v11 = dmde
    v12 = 0.0
    v13 = dlde / dvbec
    v21 = dmde * dla / dvbec - dlde * dma / dvbec
    v22 = dmde
    v23 = -dmq * dlde / dvbec
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
        [am + dmq - dla / dvbec, bm + dma + dmq * dla / dvbec, cm],
        dtype=float,
    )
    gaingam = np.linalg.inv(dp) @ dd
    dum33 = aa - np.outer(bb, gaingam)
    dum3 = np.linalg.inv(dum33) @ bb
    hh = np.array([0.0, 0.0, 1.0], dtype=float)
    gainff = -1.0 / (hh @ dum3)

    thtc = gainff * thtvdcomx * RAD
    qqf = gaingam[0] * qqcx * RAD
    thtbgf = gaingam[1] * thtbdcx * RAD
    thtugf = gaingam[2] * thtvdcx * RAD
    delec = thtc - (qqf + thtbgf + thtugf)
    delecx = delec * DEG
    return delecx, gaingam, gainff


def test_control_gamma_climb_poles_thtvdcomx_zero_elevator_finite_matches_cadac():
    vehicle, ctrl = _ready()
    store = vehicle.store
    assert store.get("pgam") == PGAM
    assert store.get("wgam") == WGAM
    assert store.get("zgam") == ZGAM
    assert store.get("dvbec") != 0.0
    assert store.get("dla") != 0.0
    assert store.get("dmde") != 0.0
    assert "time" not in store.names()
    assert "pdynmc" not in store.names()
    assert "thtvlcomx" not in store.names()
    want, want_gaingam, want_gainff = _control_gamma(store, THTVDCOMX)
    got = ctrl.control_gamma(vehicle, THTVDCOMX)
    assert isfinite(got)
    assert _approx(got, want)
    np.testing.assert_allclose(store.get("GAINGAM"), want_gaingam, rtol=RTOL, atol=ATOL)
    assert store.get("GAINGAM").shape == (3,)
    assert _approx(store.get("gainff"), want_gainff)
    assert want_gainff != 0.0
    assert np.any(want_gaingam != 0.0)
    assert want != 0.0
    assert store.get("delecx") == 0.0


def test_control_gamma_thtvdcomx_ten_matches_cadac_and_differs_from_zero():
    vehicle, ctrl = _ready()
    store = vehicle.store
    want_zero, _, _ = _control_gamma(store, THTVDCOMX)
    want, want_gaingam, want_gainff = _control_gamma(store, THTVDCOMX_EVENT)
    got = ctrl.control_gamma(vehicle, THTVDCOMX_EVENT)
    assert isfinite(got)
    assert _approx(got, want)
    np.testing.assert_allclose(store.get("GAINGAM"), want_gaingam, rtol=RTOL, atol=ATOL)
    assert _approx(store.get("gainff"), want_gainff)
    assert want != pytest.approx(want_zero, rel=RTOL, abs=ATOL)
    assert store.get("delecx") == 0.0


def test_control_gamma_does_not_write_delecx():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.set("delecx", 7.0)
    ctrl.control_gamma(vehicle, THTVDCOMX)
    assert store.get("delecx") == 7.0


def test_execute_is_pass_until_maut_dispatcher():
    vehicle, ctrl = _ready()
    store = vehicle.store
    store.set("maut", 24)
    store.set("thtvdcomx", THTVDCOMX_EVENT)
    want, _gaingam, _gainff = _control_gamma(store, THTVDCOMX_EVENT)
    assert want != 0.0
    ctrl.execute(vehicle, _ctx())
    assert store.get("delecx") == 0.0
    assert store.get("gainff") == 0.0
    np.testing.assert_array_equal(store.get("GAINGAM"), np.zeros(3))
    assert store.get("delacx") == 0.0
    assert store.get("delrcx") == 0.0


def test_control_gamma_reads_ins_qqcx_thtbdcx_thtvdcx_dvbec():
    vehicle, ctrl = _ready(qqcx=5.0, thtbdcx=4.0, thtvdcx=1.0, dvbec=800.0, dvbe=180.0)
    vehicle.store.define(Field("qqx", 40.0, "real", "out", "euler"))
    vehicle.store.define(Field("thtblx", 45.0, "real", "diag", "kinematics"))
    vehicle.store.define(Field("thtvlx", 20.0, "real", "out", "newton"))
    want, want_gaingam, want_gainff = _control_gamma(vehicle.store, THTVDCOMX)
    got = ctrl.control_gamma(vehicle, THTVDCOMX)
    assert _approx(got, want)
    np.testing.assert_allclose(
        vehicle.store.get("GAINGAM"), want_gaingam, rtol=RTOL, atol=ATOL
    )
    assert _approx(vehicle.store.get("gainff"), want_gainff)
    decoy_store = vehicle.store
    decoy_dvbe = 180.0
    dla = decoy_store.get("dla")
    dlde = decoy_store.get("dlde")
    dma = decoy_store.get("dma")
    dmq = decoy_store.get("dmq")
    dmde = decoy_store.get("dmde")
    pgam = decoy_store.get("pgam")
    wgam = decoy_store.get("wgam")
    zgam = decoy_store.get("zgam")
    aa = np.array(
        [
            [dmq, dma, -dma],
            [1.0, 0.0, 0.0],
            [0.0, dla / decoy_dvbe, -dla / decoy_dvbe],
        ],
        dtype=float,
    )
    bb = np.array([dmde, 0.0, dlde / decoy_dvbe], dtype=float)
    am = 2.0 * zgam * wgam + pgam
    bm = wgam * wgam + 2.0 * zgam * wgam * pgam
    cm = wgam * wgam * pgam
    v21 = dmde * dla / decoy_dvbe - dlde * dma / decoy_dvbe
    dp = np.array(
        [
            [dmde, 0.0, dlde / decoy_dvbe],
            [v21, dmde, -dmq * dlde / decoy_dvbe],
            [0.0, v21, v21],
        ],
        dtype=float,
    )
    dd = np.array(
        [am + dmq - dla / decoy_dvbe, bm + dma + dmq * dla / decoy_dvbe, cm],
        dtype=float,
    )
    gaingam = np.linalg.inv(dp) @ dd
    dum3 = np.linalg.inv(aa - np.outer(bb, gaingam)) @ bb
    gainff = -1.0 / (np.array([0.0, 0.0, 1.0]) @ dum3)
    thtc = gainff * THTVDCOMX * RAD
    qqf = gaingam[0] * 40.0 * RAD
    thtbgf = gaingam[1] * 45.0 * RAD
    thtugf = gaingam[2] * 20.0 * RAD
    decoy = (thtc - (qqf + thtbgf + thtugf)) * DEG
    assert got != pytest.approx(decoy, rel=RTOL, abs=ATOL)


def test_control_gamma_dvbec_zero_falls_back_to_dvbe():
    vehicle, ctrl = _ready(dvbec=0.0, dvbe=1000.0)
    want, want_gaingam, want_gainff = _control_gamma(vehicle.store, THTVDCOMX)
    got = ctrl.control_gamma(vehicle, THTVDCOMX)
    assert isfinite(got)
    assert _approx(got, want)
    np.testing.assert_allclose(
        vehicle.store.get("GAINGAM"), want_gaingam, rtol=RTOL, atol=ATOL
    )
    assert _approx(vehicle.store.get("gainff"), want_gainff)
    assert want != 0.0
    assert vehicle.store.get("dvbec") == 0.0
    assert vehicle.store.get("dvbe") == 1000.0

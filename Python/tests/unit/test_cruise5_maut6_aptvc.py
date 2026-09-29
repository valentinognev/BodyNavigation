"""CRUISE5 Fortran C2 MAUTP=6 pitch-rate + APTVC aero/TVC mix."""

import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round3.cruise5.control import Cruise5Control

RTOL = 1e-12
ATOL = 1e-14
INT_STEP = 0.05

# Planted detailed-rate-loop inputs (TR=0 → APTVC path)
APTVC = 0.5
GA = 0.4
GP = 0.3
AIZ = 1200.0
CMDEL = -0.8
RLENG = 0.5
PARM = 1.2
FLPLIM = 0.4
TVCLIM = 0.25
ALPLIM = 0.2
CNALP = 8.5
AREA = 0.929
PDYNMC = 5000.0
FTHALT = 1500.0
AMASS = 1000.0
DVBA = 200.0
WQC = 0.15
RATEP0 = 0.05
ALP0 = 0.02
TR = 0.0
MTURN = 1
MAUT = 66


def _ctx():
    return SimContext(0.0, INT_STEP, 0.0, 0.0, None, 0)


def _expected_delq_eta(*, eratep, aptvc, ga, gp, flplim, tvclim, fthalt):
    """Fortran C2PITCH detailed rate loop DELQ/ETA (pre-limit then limit)."""
    delq = -eratep * (1.0 - aptvc) * ga
    if delq > flplim:
        delq = flplim
    if delq < -flplim:
        delq = -flplim
    if fthalt > 0:
        eta = -eratep * aptvc * gp
    else:
        eta = 0.0
    if eta > tvclim:
        eta = tvclim
    if eta < -tvclim:
        eta = -tvclim
    return delq, eta


def _expected_poles(*, aptvc, ga, gp, cmom, aiz, fthalt, parm):
    """Fortran POLEA / POLEP."""
    polea = (1.0 - aptvc) * ga * cmom / aiz
    polep = aptvc * gp * fthalt * parm / aiz
    return polea, polep


def _ready(**overrides):
    vehicle = type("V", (), {"store": StateStore()})()
    control = Cruise5Control()
    control.define(vehicle)
    store = vehicle.store
    for name, value, ftype in (
        ("tgv", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), "mat"),
        ("pdynmc", PDYNMC, "real"),
        ("area", AREA, "real"),
        ("thrust", FTHALT, "real"),
        ("mass", AMASS, "real"),
        ("dvbe", DVBA, "real"),
    ):
        if name not in store.names():
            store.define(Field(name, value, ftype, "out", "test"))
        else:
            store.set(name, value)
    store.set("maut", MAUT)
    store.set("mturn", MTURN)
    store.set("mcontrol", 0)
    store.set("aptvc", APTVC)
    store.set("ga", GA)
    store.set("gp", GP)
    store.set("aiz", AIZ)
    store.set("cmdel", CMDEL)
    store.set("rleng", RLENG)
    store.set("parm", PARM)
    store.set("flplim", FLPLIM)
    store.set("tvclim", TVCLIM)
    store.set("alplim", ALPLIM)
    store.set("cnalp", CNALP)
    store.set("tr", TR)
    store.set("wqc", WQC)
    store.set("wpc", 0.0)
    store.set("ratep", RATEP0)
    store.set("ratepd", 0.0)
    store.set("alp", ALP0)
    store.set("alpd", 0.0)
    store.set("tphi", 0.4)
    store.set("philim", 1.22)
    for key, value in overrides.items():
        store.set(key, value)
    return vehicle, control


def test_mautp6_aptvc_splits_delq_eta_per_fortran():
    """MAUTP=6 rate law; APTVC=0.5 splits DELQ/ETA per POLEA/POLEP formulas."""
    vehicle, control = _ready()
    store = vehicle.store
    control.execute(vehicle, _ctx())

    eratep = WQC - RATEP0
    want_delq, want_eta = _expected_delq_eta(
        eratep=eratep,
        aptvc=APTVC,
        ga=GA,
        gp=GP,
        flplim=FLPLIM,
        tvclim=TVCLIM,
        fthalt=FTHALT,
    )
    cmom = PDYNMC * AREA * RLENG * CMDEL
    want_polea, want_polep = _expected_poles(
        aptvc=APTVC,
        ga=GA,
        gp=GP,
        cmom=cmom,
        aiz=AIZ,
        fthalt=FTHALT,
        parm=PARM,
    )

    np.testing.assert_allclose(store.get("delq"), want_delq, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("eta"), want_eta, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("polea"), want_polea, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("polep"), want_polep, rtol=RTOL, atol=ATOL)
    # Equal APTVC mix with equal |gains| would share; with GA!=GP, aero share is (1-aptvc)*GA
    assert abs(want_delq) > 0.0
    assert abs(want_eta) > 0.0
    np.testing.assert_allclose(
        want_delq / (-eratep * GA),
        1.0 - APTVC,
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        want_eta / (-eratep * GP),
        APTVC,
        rtol=RTOL,
        atol=ATOL,
    )
    assert np.isfinite(store.get("alphax"))
    assert store.get("TBV").shape == (3, 3)


def test_mautp6_eta_zero_when_thrust_zero():
    vehicle, control = _ready(thrust=0.0)
    store = vehicle.store
    control.execute(vehicle, _ctx())
    assert store.get("eta") == 0.0
    eratep = WQC - RATEP0
    want_delq, _ = _expected_delq_eta(
        eratep=eratep,
        aptvc=APTVC,
        ga=GA,
        gp=GP,
        flplim=FLPLIM,
        tvclim=TVCLIM,
        fthalt=0.0,
    )
    np.testing.assert_allclose(store.get("delq"), want_delq, rtol=RTOL, atol=ATOL)


def test_mcontrol_path_unchanged_when_maut_zero():
    """Task 22 mcontrol==6 altitude+load still runs when maut defaults to 0."""
    vehicle = type("V", (), {"store": StateStore()})()
    control = Cruise5Control()
    control.define(vehicle)
    store = vehicle.store
    store.define(Field(
        "tgv", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "mat", "init", "newton",
    ))
    for name, value, ftype in (
        ("FSPV", np.array([2.0, 0.0, -12.0]), "vec"),
        ("grav", 9.81, "real"),
        ("mass", 1000.0, "real"),
        ("dvbe", 200.0, "real"),
        ("pdynmc", 5000.0, "real"),
        ("thrust", 1500.0, "real"),
        ("area", 0.929, "real"),
        ("cla", 0.11, "real"),
        ("vbeg", np.array([200.0, 0.0, 0.0]), "vec"),
        ("alt", 7000.0, "real"),
    ):
        if name not in store.names():
            store.define(Field(name, value, ftype, "out", "test"))
        else:
            store.set(name, value)
    store.set("maut", 0)
    store.set("mcontrol", 6)
    store.set("anposlimx", 3.0)
    store.set("anneglimx", -1.0)
    store.set("alpposlimx", 15.0)
    store.set("alpneglimx", -10.0)
    store.set("gacp", 10.0)
    store.set("ta", 0.8)
    store.set("altcom", 7000.0)
    store.set("gh", 0.3)
    store.set("gv", 1.0)
    store.set("altdlim", 50.0)
    control.execute(vehicle, _ctx())
    assert store.get("phimvx") == 0.0
    assert np.isfinite(store.get("alphax"))

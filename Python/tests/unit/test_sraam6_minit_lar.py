"""SRAAM6 G1I — MINIT LAR/CIRCLE auto-geometry (Fortran MODULE.FOR G1I)."""

import math

import numpy as np

from cadac.constants import DEG
from cadac.kernel.executive import SimContext
from cadac.kernel.state import StateStore
from cadac.vehicles.flat6.sraam6.target import Sraam6TargetMinit

RTOL = 1e-12
ATOL = 1e-12


def _matcar(dbe, azbel, elbel):
    celb = math.cos(elbel)
    selb = math.sin(elbel)
    cazb = math.cos(azbel)
    sazb = math.sin(azbel)
    return np.array([dbe * celb * cazb, dbe * celb * sazb, -dbe * selb])


def _fortran_minit21_geometry():
    """Hand-computed G1I LAR-1 (MINC=2, MINM=1) — alpha before ST2EL set."""
    an1c = 1.0
    dvt1e = 240.0
    ht1e = 5000.0
    tauhx = 60.0
    dvt2e = 240.0
    an2c = 1.0
    sighx = 20.0
    wloadt2 = 3247.0
    clat2 = 0.0523
    ht2e = 5000.0
    rhl = 10000.0
    # First alpha uses ST2EL(3)==0 (Fortran order)
    rhot2 = 1.225 * (1.0 + 0.0 / 41900.0) ** 4
    alpt2x = 2.0 * an2c / (rhot2 * dvt2e**2) * wloadt2 / clat2
    st2el = np.array(
        [
            rhl * math.cos(tauhx / DEG),
            rhl * math.sin(tauhx / DEG),
            -ht2e,
        ]
    )
    thtt2lx = 0.0
    phit2lx = 0.0
    psit2lx = -180.0 + tauhx - sighx
    alamhx = sighx - alpt2x
    st1el = np.array([0.0, 0.0, -ht1e])
    psit1lx = 0.0
    thtt1lx = 0.0
    phit1lx = 0.0
    anuhx = psit1lx - psit2lx
    vt2el = _matcar(dvt2e, psit2lx / DEG, thtt2lx / DEG)
    vt1el = _matcar(dvt1e, psit1lx / DEG, thtt1lx / DEG)
    return {
        "ST2EL": st2el,
        "ST1EL": st1el,
        "VT2EL": vt2el,
        "VT1EL": vt1el,
        "alpt2x": alpt2x,
        "alamhx": alamhx,
        "tauhx": tauhx,
        "sighx": sighx,
        "psit2lx": psit2lx,
        "thtt2lx": thtt2lx,
        "phit2lcx": phit2lx,
        "psit1lx": psit1lx,
        "thtt1lx": thtt1lx,
        "phit1lcx": phit1lx,
        "anuhx": anuhx,
        "an1c": an1c,
    }


def _fortran_minit13_geometry():
    """Hand-computed G1I LAR-3 (MINC=1, MINM=3) — shooter-centered one-circle."""
    an1c = 7.5
    dvt1e = 240.0
    ht1e = 5000.0
    dvt2e = 240.0
    an2c = 7.572
    wloadt2 = 3249.0
    clat2 = 0.0523
    ht2e = 5000.0
    alamhx = 30.0
    rhl = 4000.0
    rhot2 = 1.225 * (1.0 + 0.0 / 41900.0) ** 4
    alpt2x = 2.0 * an2c / (rhot2 * dvt2e**2) * wloadt2 / clat2
    st2el = np.array([0.0, 0.0, -ht2e])
    phit2l = math.atan(an2c)
    sphit2 = math.sin(phit2l)
    phit2lx = DEG * phit2l
    psit2lx = -DEG * math.atan(sphit2 * math.tan(alpt2x / DEG))
    thtt2lx = 0.0
    st1el = np.array(
        [
            rhl * math.cos(alamhx / DEG),
            rhl * math.sin(alamhx / DEG),
            -ht1e,
        ]
    )
    thtt1lx = 0.0
    psit1lx = alpt2x + 2.0 * alamhx - 180.0
    phit1lx = -DEG * math.atan(an1c)
    tauhx = -(alpt2x + alamhx)
    sighx = -tauhx
    anuhx = psit1lx - psit2lx
    vt2el = _matcar(dvt2e, psit2lx / DEG, thtt2lx / DEG)
    vt1el = _matcar(dvt1e, psit1lx / DEG, thtt1lx / DEG)
    return {
        "ST2EL": st2el,
        "ST1EL": st1el,
        "VT2EL": vt2el,
        "VT1EL": vt1el,
        "alpt2x": alpt2x,
        "alamhx": alamhx,
        "tauhx": tauhx,
        "sighx": sighx,
        "psit2lx": psit2lx,
        "thtt2lx": thtt2lx,
        "phit2lcx": phit2lx,
        "psit1lx": psit1lx,
        "thtt1lx": thtt1lx,
        "phit1lcx": phit1lx,
        "anuhx": anuhx,
        "an1c": an1c,
    }


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(0.0, 0.00123, 0.0, 0.0, None, 0)


def _ready_21():
    vehicle = _Vehicle()
    mod = Sraam6TargetMinit()
    mod.define(vehicle)
    store = vehicle.store
    store.set("minit", 21)
    store.set("an1c", 1.0)
    store.set("dvt1e", 240.0)
    store.set("ht1e", 5000.0)
    store.set("tauhx", 60.0)
    store.set("dvt2e", 240.0)
    store.set("an2c", 1.0)
    store.set("sighx", 20.0)
    store.set("wloadt2", 3247.0)
    store.set("clat2", 0.0523)
    store.set("ht2e", 5000.0)
    store.set("rhl", 10000.0)
    mod.initialize(vehicle, _ctx())
    return store


def _ready_13():
    vehicle = _Vehicle()
    mod = Sraam6TargetMinit()
    mod.define(vehicle)
    store = vehicle.store
    store.set("minit", 13)
    store.set("an1c", 7.5)
    store.set("dvt1e", 240.0)
    store.set("ht1e", 5000.0)
    store.set("dvt2e", 240.0)
    store.set("an2c", 7.572)
    store.set("wloadt2", 3249.0)
    store.set("clat2", 0.0523)
    store.set("ht2e", 5000.0)
    store.set("alamhx", 30.0)
    store.set("rhl", 4000.0)
    mod.initialize(vehicle, _ctx())
    return store


def test_name_is_target():
    assert Sraam6TargetMinit().name == "target"


def test_minit_21_lar1_geometry_matches_fortran():
    """Planted RHL/TAUHX/… → ST1EL/ST2EL/SBEL/aspect match G1I MINIT=21."""
    store = _ready_21()
    exp = _fortran_minit21_geometry()
    np.testing.assert_allclose(store.get("ST2EL"), exp["ST2EL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ST1EL"), exp["ST1EL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VT2EL"), exp["VT2EL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VT1EL"), exp["VT1EL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBEL"), exp["ST2EL"], rtol=RTOL, atol=ATOL)
    for key in (
        "psit2lx",
        "thtt2lx",
        "alamhx",
        "alpt2x",
        "tauhx",
        "sighx",
        "phit1lcx",
        "phit2lcx",
        "psit1lx",
        "thtt1lx",
        "anuhx",
    ):
        np.testing.assert_allclose(store.get(key), exp[key], rtol=RTOL, atol=ATOL)


def test_minit_13_lar3_geometry_matches_fortran():
    """Planted ALAMHX/RHL/… → ST1EL/ST2EL/SBEL/aspect match G1I MINIT=13."""
    store = _ready_13()
    exp = _fortran_minit13_geometry()
    np.testing.assert_allclose(store.get("ST2EL"), exp["ST2EL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ST1EL"), exp["ST1EL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VT2EL"), exp["VT2EL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VT1EL"), exp["VT1EL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBEL"), exp["ST2EL"], rtol=RTOL, atol=ATOL)
    for key in (
        "psit2lx",
        "thtt2lx",
        "alamhx",
        "alpt2x",
        "tauhx",
        "sighx",
        "phit1lcx",
        "phit2lcx",
        "psit1lx",
        "thtt1lx",
        "anuhx",
    ):
        np.testing.assert_allclose(store.get(key), exp[key], rtol=RTOL, atol=ATOL)

"""SRAAM5 G1I target — MTARG=21 LAR-1 geometry (Fortran MODULE.FOR G1I)."""

import math

import numpy as np

from cadac.constants import DEG
from cadac.kernel.executive import SimContext
from cadac.kernel.state import StateStore
from cadac.vehicles.flat5.sraam5.target import Sraam5Target

RTOL = 1e-12
ATOL = 1e-12

# Planted MTARG=21 fields from cases/sraam5/inlar1.jsonc
MTARG = 21
AN1C = 1.0
DVT1E = 240.0
HT1E = 5000.0
TAUHX = 10.0
DVT2E = 240.0
AN2C = 1.0
SIGHX = 20.0
WLOADT2 = 3247.0
CLAT2 = 0.0523
HT2E = 5000.0
RHL = 6000.0


def _fortran_mtarg21_geometry():
    """Hand-computed G1I LAR-1 (MTARGC=2, MTARGM=1) init math."""
    st2el = np.array(
        [
            RHL * math.cos(TAUHX / DEG),
            RHL * math.sin(TAUHX / DEG),
            -HT2E,
        ]
    )
    rhot2 = 1.225 * (1.0 + st2el[2] / 41900.0) ** 4
    alpt2x = 2.0 * AN2C / (rhot2 * DVT2E**2) * WLOADT2 / CLAT2
    psit2lx = -180.0 + TAUHX - SIGHX
    alamhx = SIGHX - alpt2x
    thtt2lx = 0.0
    phit2lx = 0.0
    st1el = np.array([0.0, 0.0, -HT1E])
    psit1lx = 0.0
    thtt1lx = 0.0
    phit1lx = DEG * math.atan(AN1C)
    if AN1C <= 1.0:
        phit1lx = 0.0
    anuhx = psit1lx - psit2lx
    psit2l = psit2lx / DEG
    thtt2l = thtt2lx / DEG
    vt2el = np.array(
        [
            DVT2E * math.cos(thtt2l) * math.cos(psit2l),
            DVT2E * math.cos(thtt2l) * math.sin(psit2l),
            -DVT2E * math.sin(thtt2l),
        ]
    )
    psit1l = psit1lx / DEG
    thtt1l = thtt1lx / DEG
    vt1el = np.array(
        [
            DVT1E * math.cos(thtt1l) * math.cos(psit1l),
            DVT1E * math.cos(thtt1l) * math.sin(psit1l),
            -DVT1E * math.sin(thtt1l),
        ]
    )
    return {
        "ST2EL": st2el,
        "ST1EL": st1el,
        "VT2EL": vt2el,
        "VT1EL": vt1el,
        "alpt2x": alpt2x,
        "alamhx": alamhx,
        "psit2lx": psit2lx,
        "thtt2lx": thtt2lx,
        "phit2lcx": phit2lx,
        "psit1lx": psit1lx,
        "thtt1lx": thtt1lx,
        "phit1lcx": phit1lx,
        "anuhx": anuhx,
    }


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(0.0, 0.0123, 0.0, 0.0, None, 0)


def _ready():
    vehicle = _Vehicle()
    target = Sraam5Target()
    target.define(vehicle)
    store = vehicle.store
    store.set("mtarg", MTARG)
    store.set("an1c", AN1C)
    store.set("dvt1e", DVT1E)
    store.set("ht1e", HT1E)
    store.set("tauhx", TAUHX)
    store.set("dvt2e", DVT2E)
    store.set("an2c", AN2C)
    store.set("sighx", SIGHX)
    store.set("wloadt2", WLOADT2)
    store.set("clat2", CLAT2)
    store.set("ht2e", HT2E)
    store.set("rhl", RHL)
    target.initialize(vehicle, _ctx())
    return store


def test_name_is_target():
    assert Sraam5Target().name == "target"


def test_mtarg_21_init_geometry_matches_fortran():
    """Planted RHL/HT1E/… → ST1EL/ST2EL/angles match G1I MTARG=21."""
    store = _ready()
    exp = _fortran_mtarg21_geometry()
    np.testing.assert_allclose(store.get("ST2EL"), exp["ST2EL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ST1EL"), exp["ST1EL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VT2EL"), exp["VT2EL"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VT1EL"), exp["VT1EL"], rtol=RTOL, atol=ATOL)
    for key in (
        "psit2lx",
        "thtt2lx",
        "alamhx",
        "alpt2x",
        "phit1lcx",
        "phit2lcx",
        "psit1lx",
        "thtt1lx",
        "anuhx",
    ):
        np.testing.assert_allclose(store.get(key), exp[key], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBEL"), exp["ST2EL"], rtol=RTOL, atol=ATOL)

"""CRUISE5 Fortran C1 MROLL=1 inverted bank command (PHIBVC=3.1412-PHIBVC)."""

import numpy as np

from cadac.constants import RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.earth import cadine, cadtei, cadtge
from cadac.math.frames import mat2tr
from cadac.vehicles.round3.cruise5.guidance import Cruise5Guidance

RTOL = 1e-12
ATOL = 1e-14

# INPUT.ASC bank-to-turn roll gain block
GGP = 3.0
BGP = 1.0
ALDEAD = 0.01740
PHILIM = 1.22  # rad
AGRAV = 9.81
# Planted FSPCB(3) negative (level flight-ish) so SIGN is well-defined
FSPCB = (0.0, 0.0, -AGRAV)
MTURN = 1
# Fortran literal for inverted attitude
PI_INV = 3.1412


def _ctx():
    return SimContext(0.0, 0.05, 0.0, 0.0, None, 0)


def _round3_state(lonx, latx, alt, psivgx, thtvgx, dvbe, time):
    tei = cadtei(time)
    tge = cadtge(lonx * RAD, latx * RAD)
    tig = tei.T @ tge.T
    sbii = cadine(lonx * RAD, latx * RAD, alt, time)
    vbeg = mat2tr(psivgx * RAD, thtvgx * RAD).T @ np.array([dvbe, 0.0, 0.0])
    return tig, sbii, vbeg


def _ready(mguidance):
    """Same geometry; MGUID=|MROLL|MGUIDL|MGUIDP| with MGUIDL=3 lateral line."""
    vehicle = type("V", (), {"store": StateStore()})()
    guidance = Cruise5Guidance()
    guidance.define(vehicle)
    store = vehicle.store
    tig, sbii, vbeg = _round3_state(14.7, 35.4, 7000.0, 90.0, 0.0, 200.0, 0.0)
    for name, value, ftype in (
        ("time", 0.0, "real"),
        ("grav", AGRAV, "real"),
        ("tig", tig, "mat"),
        ("thtvgx", 0.0, "real"),
        ("vbeg", vbeg, "vec"),
        ("sbii", sbii, "vec"),
        ("philimx", 70.0, "real"),
        ("philim", PHILIM, "real"),
        ("anposlimx", 10.0, "real"),
        ("anneglimx", -10.0, "real"),
        ("allimx", 10.0, "real"),
        ("phicx", 0.0, "real"),
        ("alcomx", 0.0, "real"),
        ("ancomx", 0.0, "real"),
        ("FSPCB", FSPCB, "vec"),
        ("mturn", MTURN, "int"),
        ("ggp", GGP, "real"),
        ("bgp", BGP, "real"),
        ("aldead", ALDEAD, "real"),
    ):
        if name not in store:
            store.define(Field(name, value, ftype, "out", "test"))
        else:
            store.set(name, value)
    store.set("mguidance", mguidance)
    # Offset waypoint → nonzero lateral ACV2 / PHIBVC
    store.set("wp_lonx", 14.9)
    store.set("wp_latx", 35.4)
    store.set("wp_alt", 0.0)
    store.set("psifgx", 90.0)
    store.set("thtfgx", 0.0)
    store.set("line_gain", 1.0)
    store.set("nl_gain_fact", 0.6)
    store.set("decrement", 1000.0)
    store.set("point_gain", 1.0)
    return vehicle, guidance


def test_mroll1_inverts_phibvc_vs_mroll0_same_geometry():
    """Fortran C1: IF(MROLL.EQ.1) PHIBVC=3.1412-PHIBVC (bank-to-turn MGUIDL=3)."""
    v0, g0 = _ready(mguidance=33)  # MROLL=0 MGUIDL=3 MGUIDP=3
    g0.execute(v0, _ctx())
    phibvc0 = v0.store.get("phibvc")
    assert v0.store.get("mroll") == 0
    assert abs(phibvc0) > 1e-6  # nonzero bank for offset waypoint

    v1, g1 = _ready(mguidance=133)  # MROLL=1 same MGUIDL/MGUIDP
    g1.execute(v1, _ctx())
    phibvc1 = v1.store.get("phibvc")
    assert v1.store.get("mroll") == 1
    assert v1.store.get("mguidl") == 3
    assert v1.store.get("mguidp") == 3

    np.testing.assert_allclose(phibvc1, PI_INV - phibvc0, rtol=RTOL, atol=ATOL)
    # Lateral load command unchanged by MROLL (inversion is on PHIBVC only)
    np.testing.assert_allclose(
        v1.store.get("alcomx"), v0.store.get("alcomx"), rtol=RTOL, atol=ATOL
    )


def test_mroll0_phibvc_matches_fortran_gain_formula():
    """PHIBVC=-SIGN(PC,FSPCB(3))*DUM limited by PHILIM (MROLL=0)."""
    vehicle, guidance = _ready(mguidance=33)
    store = vehicle.store
    algv = guidance.guidance_line(vehicle)
    acv2 = float(algv[1])
    dum = acv2 / AGRAV
    if abs(dum) < ALDEAD:
        dum = 0.0
    fspcb3 = FSPCB[2]
    pc = GGP / (0.01 + BGP + abs(fspcb3 / AGRAV))
    sign_pc = pc if fspcb3 >= 0.0 else -pc
    want = -sign_pc * dum
    if want > PHILIM:
        want = PHILIM
    if want < -PHILIM:
        want = -PHILIM

    guidance.execute(vehicle, _ctx())
    np.testing.assert_allclose(store.get("phibvc"), want, rtol=RTOL, atol=ATOL)

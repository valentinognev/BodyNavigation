"""CRUISE5 Fortran C1 MGUIDP=2 look-fwd TF/OA + D1TER obstacle draws."""

from math import log, sqrt

import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round3.cruise5.guidance import Cruise5Guidance

RTOL = 1e-12
ATOL = 1e-14
INT_STEP = 0.05

# INPUT.ASC TF/OA block (verbatim scales)
DCELL = 100.0
RAHEAD = 3000.0
DHTRC = 10.0
SLOPE = 0.05
OCCDEN = 0.0005
SIGOBS = 10.0
ISEED2 = 12345
TLEAD = 3.0
GELEV = 0.07
RACQ = 1200.0


def _ran_ms(iseed):
    """Microsoft FORTRAN RAN(ISEED) — shared formula with guidance port."""
    iseed = int(iseed) * 65539
    if iseed < 0:
        iseed = (iseed + 2147483647) + 1
    # Keep 32-bit signed wrap identical to typical MS runtime
    iseed = ((iseed + 2**31) % 2**32) - 2**31
    if iseed < 0:
        iseed = (iseed + 2147483647) + 1
    u = float(iseed) * 0.4656613e-9
    return u, iseed


def _ctx():
    return SimContext(0.0, INT_STEP, 0.0, 0.0, None, 0)


def _expected_obstacle(iseed2, gndtck, occden, sigobs):
    """Fortran C1 obstacle draw: exponential wait + Rayleigh height."""
    u1, iseed2 = _ran_ms(iseed2)
    docc = -log(1.0 - u1) / occden
    gndocc = gndtck + docc
    u2, iseed2 = _ran_ms(iseed2)
    rayl = sqrt(-2.0 * log(1.0 - u2))
    dhobst = rayl * sigobs
    return dhobst, gndocc, iseed2


def _expected_elmax(ho, hbe, thtvl, dhtrc, dcell, n):
    """Fortran C1 max elevation within RAHEAD (final value after DO 20)."""
    elm = -99999.0
    elmax = 0.0
    for i in range(2, n + 1):
        el_i = (ho[i] + dhtrc - hbe) / ((i - 1) * dcell)
        if el_i > elm:
            elm = el_i
        elmax = elm - thtvl
    return elmax


def _ready_look_fwd(**overrides):
    """MGUID=32 → MGUIDP=2; plant full terrain stack ready for OA/TF step."""
    vehicle = type("V", (), {"store": StateStore()})()
    guidance = Cruise5Guidance()
    guidance.define(vehicle)
    store = vehicle.store

    n = int(round(RAHEAD / DCELL))
    k = int(round(RACQ / DCELL)) + 1
    # Plant HO/H: flat terrain 100 m, one tall obstacle at sensor cell K
    ho = [0.0] * (n + 1)
    h = [0.0] * (n + 1)
    for i in range(1, n + 1):
        ho[i] = 100.0
        h[i] = 100.0
    ho[k] = 100.0 + 200.0  # tall obstacle → ELMAX > SLOPE after stack shift

    for name, value, ftype in (
        ("thtvgx", 0.0, "real"),
        ("dvbe", 272.0, "real"),
        ("time", 5.0, "real"),
        ("philimx", 70.0, "real"),
        ("anposlimx", 10.0, "real"),
        ("anneglimx", -10.0, "real"),
        ("allimx", 10.0, "real"),
        ("grav", 9.80665, "real"),
        ("phicx", 0.0, "real"),
        ("ancomx", 0.0, "real"),
        ("alcomx", 0.0, "real"),
        ("tig", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), "mat"),
        ("sbii", (0.0, 0.0, -150.0), "vec"),
        ("vbeg", (272.0, 0.0, 0.0), "vec"),
    ):
        if name not in store:
            store.define(Field(name, value, ftype, "out", "test"))
        else:
            store.set(name, value)

    store.set("mguidance", 32)
    store.set("dcell", DCELL)
    store.set("rahead", RAHEAD)
    store.set("dhtrc", DHTRC)
    store.set("slope", SLOPE)
    store.set("occden", OCCDEN)
    store.set("sigobs", SIGOBS)
    store.set("iseed2", ISEED2)
    store.set("tlead", TLEAD)
    store.set("gelev", GELEV)
    store.set("racq", RACQ)
    store.set("hbe", 150.0)
    store.set("thtvl", 0.0)
    store.set("gndtck", 4000.0)
    store.set("gndpt", 3900.0)  # DGND = 100 >= DCELL
    store.set("j_ter", n)  # past init
    store.set("hge", 100.0)
    store.set("gndocc", 0.0)  # force obstacle draw (GNDTCK >= GNDOCC)
    store.set("maut", 0)
    store.set("wp_lonx", 14.7)
    store.set("wp_latx", 35.4)
    store.set("wp_alt", 0.0)
    store.set("psifgx", 0.0)
    store.set("thtfgx", 0.0)
    store.set("line_gain", 2.0)
    store.set("nl_gain_fact", 0.6)
    store.set("decrement", 2000.0)

    guidance.plant_terrain_stack(ho, h)
    for key, value in overrides.items():
        store.set(key, value)
    return vehicle, guidance, n, k, ho


def _nint(value):
    """Fortran NINT — nearest integer, halves away from zero."""
    if value >= 0:
        return int(value + 0.5)
    return int(value - 0.5)


def test_mguidp2_oa_elmax_sets_ancom_per_fortran():
    """MGUIDP=2, planted HO → ELMAX > SLOPE → ANCOM = 1 + GELEV*ELMAX*DVBE."""
    vehicle, guidance, n, k, ho = _ready_look_fwd(gndocc=1.0e99)  # suppress new draw
    store = vehicle.store
    hbe = store.get("hbe")
    thtvl = store.get("thtvl")
    hge = store.get("hge")
    # Fortran shifts stack before ELMAX: HO(i)=HO(i+1), HO(N)=HGE
    ho_shift = [0.0] * (n + 1)
    for i in range(1, n):
        ho_shift[i] = ho[i + 1]
    ho_shift[n] = hge
    elmax_want = _expected_elmax(ho_shift, hbe, thtvl, DHTRC, DCELL, n)
    assert elmax_want > SLOPE
    ancom_want = 1.0 + GELEV * elmax_want * store.get("dvbe")

    guidance.execute(vehicle, _ctx())

    np.testing.assert_allclose(store.get("elmax"), elmax_want, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ancomx"), ancom_want, rtol=RTOL, atol=ATOL)
    assert store.get("tfoa") == 9.0


def test_mguidp2_tf_elmax_sets_hbgs_per_fortran():
    """MGUIDP=2, flat HO → ELMAX < SLOPE → HBGS = HBE - H(NL), TFOA=10."""
    vehicle, guidance, n, _k, ho = _ready_look_fwd(gndocc=1.0e99)
    store = vehicle.store
    # Flatten obstacle so ELMAX < SLOPE
    for i in range(1, n + 1):
        ho[i] = 100.0
    h_flat = list(ho)
    guidance.plant_terrain_stack(ho, h_flat)
    hbe = store.get("hbe")
    thtvl = store.get("thtvl")
    hge = store.get("hge")
    ho_shift = [0.0] * (n + 1)
    h_shift = [0.0] * (n + 1)
    for i in range(1, n):
        ho_shift[i] = ho[i + 1]
        h_shift[i] = h_flat[i + 1]
    ho_shift[n] = hge
    h_shift[n] = hge
    elmax_want = _expected_elmax(ho_shift, hbe, thtvl, DHTRC, DCELL, n)
    assert elmax_want < SLOPE
    rlead = store.get("dvbe") * TLEAD
    nl = int(round(rlead / DCELL)) + 1
    hbgs_want = hbe - h_shift[nl]

    guidance.execute(vehicle, _ctx())

    np.testing.assert_allclose(store.get("elmax"), elmax_want, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("hbgs"), hbgs_want, rtol=RTOL, atol=ATOL)
    assert store.get("tfoa") == 10.0


def test_obstacle_draw_occden_sigobs_with_planted_iseed2():
    """Occurrence (OCCDEN) + Rayleigh (SIGOBS) match Fortran with planted ISEED2."""
    vehicle, guidance, n, k, ho = _ready_look_fwd()
    store = vehicle.store
    gndtck = store.get("gndtck")
    dhobst_want, gndocc_want, iseed2_want = _expected_obstacle(
        ISEED2, gndtck, OCCDEN, SIGOBS
    )

    guidance.execute(vehicle, _ctx())

    np.testing.assert_allclose(store.get("dhobst"), dhobst_want, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("gndocc"), gndocc_want, rtol=RTOL, atol=ATOL)
    assert store.get("iseed2") == iseed2_want


def test_mguidance_32_no_longer_raises():
    vehicle, guidance, *_ = _ready_look_fwd(gndocc=1.0e99)
    guidance.execute(vehicle, _ctx())
    assert vehicle.store.get("mguidance") == 32


def test_nl_persists_for_inter_cell_tf_hbgs():
    """Fortran NL is static: inter-cell TF must use H(NL), not reset NL=1."""
    vehicle, guidance, n, _k, ho = _ready_look_fwd(gndocc=1.0e99)
    store = vehicle.store
    for i in range(1, n + 1):
        ho[i] = 100.0
    h_flat = [0.0] * (n + 1)
    for i in range(1, n + 1):
        h_flat[i] = 100.0
    guidance.plant_terrain_stack(ho, h_flat)

    # Step 1: cell advance fills NL = NINT(DVBE*TLEAD/DCELL)+1
    guidance.execute(vehicle, _ctx())
    nl_want = _nint(store.get("dvbe") * TLEAD / DCELL) + 1
    assert nl_want > 1
    assert store.get("tfoa") == 10.0

    # Plant distinct H(1) vs H(NL); no further cell advance (DGND < DCELL)
    guidance._h[1] = 10.0
    guidance._h[nl_want] = 70.0
    store.set("gndtck", store.get("gndpt") + 1.0)  # DGND = 1 < DCELL
    store.set("hbe", 150.0)

    guidance.execute(vehicle, _ctx())

    # Persisted NL → HBGS = HBE - H(NL); reset-to-1 would yield 150 - 10 = 140
    hbgs_want = 150.0 - 70.0
    np.testing.assert_allclose(store.get("hbgs"), hbgs_want, rtol=RTOL, atol=ATOL)
    assert guidance._nl == nl_want

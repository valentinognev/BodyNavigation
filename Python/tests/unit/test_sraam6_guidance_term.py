from math import atan2, cos, sin, sqrt, tan

import numpy as np

from cadac.constants import AGRAV
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat3tr
from cadac.vehicles.flat6.sraam6.guidance import Sraam6Guidance

RTOL = 1e-12
ATOL = 1e-14
SMALL = 1e-7

THTPB = 0.12
PSIPB = -0.08
SIGDPY = 0.015
SIGDPZ = -0.025
FSPB = np.array([25.0, 2.0, -40.0])
GMAX = 5.0
TBL = mat3tr(0.2, -0.1, 0.3)
STEL = np.array([10000.0, 500.0, -2000.0])
SBEL = np.array([0.0, 0.0, -5000.0])
VBEL = np.array([250.0, 0.0, 0.0])
VTEL = np.array([-250.0, 0.0, 0.0])
GNAV = 3.75
TRCVEL = 10e-4


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(sim_time=0.0):
    return SimContext(
        sim_time=sim_time,
        int_step=0.001,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _cpp_term(
    *,
    thtpb,
    psipb,
    sigdpy,
    sigdpz,
    fspb,
    gmax,
    tbl,
    stel,
    sbel,
    vbel,
    vtel,
    gnav,
    time,
    trcond,
    trcvel,
):
    sbtl = sbel - stel
    dbt = float(np.sqrt(float(sbtl @ sbtl)))
    dum = float(sbtl @ (vbel - vtel))
    dcvel = abs(dum / dbt)
    if time > 3.0 and dcvel < trcvel:
        trcond = 1
    fspcb1 = float(fspb[0])
    adely = fspcb1 * tan(psipb) / AGRAV
    adelz = fspcb1 * tan(thtpb) / (cos(psipb) * AGRAV)
    gravb = tbl @ np.array([0.0, 0.0, 1.0])
    gn = gnav * dcvel
    apny = gn * sigdpz / (cos(psipb) * AGRAV)
    apnz = gn * (sigdpz * tan(thtpb) * tan(psipb) + sigdpy / cos(thtpb)) / AGRAV
    all_ = apny + adely - float(gravb[1])
    ann = apnz + adelz + float(gravb[2])
    aa = sqrt(all_ * all_ + ann * ann)
    if aa > gmax:
        aa = gmax
    if abs(ann) < SMALL and abs(all_) < SMALL:
        phi = 0.0
    else:
        phi = atan2(ann, all_)
    return {
        "dcvel": dcvel,
        "trcond": trcond,
        "gn": gn,
        "apny": apny,
        "apnz": apnz,
        "adely": adely,
        "adelz": adelz,
        "all": all_,
        "ann": ann,
        "aa": aa,
        "alcomx": aa * cos(phi),
        "ancomx": aa * sin(phi),
    }


def _plant_term(
    store,
    *,
    thtpb=THTPB,
    psipb=PSIPB,
    sigdpy=SIGDPY,
    sigdpz=SIGDPZ,
    fspb=None,
    gmax=GMAX,
    tbl=None,
    stel=STEL,
    sbel=SBEL,
    vbel=None,
    vtel=None,
    time=0.0,
    trcond=0,
    trcvel=TRCVEL,
):
    if fspb is None:
        fspb = FSPB
    if tbl is None:
        tbl = TBL
    if vbel is None:
        vbel = VBEL
    if vtel is None:
        vtel = VTEL
    store.define(Field("STEL", stel, "vec", "out", "seeker"))
    store.define(Field("VTEL", vtel, "vec", "out", "seeker"))
    store.define(Field("SBEL", sbel, "vec", "state", "newton"))
    store.define(Field("VBEL", vbel, "vec", "state", "newton"))
    store.define(Field("TBL", tbl, "mat", "out", "kinematics"))
    store.define(Field("time", time, "real", "exec", "kinematics"))
    store.define(Field("thtpb", thtpb, "real", "out", "seeker"))
    store.define(Field("psipb", psipb, "real", "out", "seeker"))
    store.define(Field("sigdpy", sigdpy, "real", "out", "seeker"))
    store.define(Field("sigdpz", sigdpz, "real", "out", "seeker"))
    store.define(Field("FSPB", fspb, "vec", "out", "newton"))
    store.define(Field("gmax", gmax, "real", "diag", "aerodynamics"))
    store.define(Field("trcond", trcond, "int", "diag", "aerodynamics"))
    store.define(Field("trcvel", trcvel, "real", "data", "aerodynamics"))


def _ready(*, mguid=6, mnav=0, gnav=GNAV, time=0.0, **plant):
    vehicle = _Vehicle()
    guidance = Sraam6Guidance()
    guidance.define(vehicle)
    _plant_term(vehicle.store, time=time, **plant)
    store = vehicle.store
    store.set("mguid", mguid)
    store.set("mnav", mnav)
    store.set("gnav", gnav)
    return vehicle, guidance, _ctx(sim_time=time)


def test_frozen_alcomx_ancomx_match_cpp():
    want = _cpp_term(
        thtpb=THTPB,
        psipb=PSIPB,
        sigdpy=SIGDPY,
        sigdpz=SIGDPZ,
        fspb=FSPB,
        gmax=GMAX,
        tbl=TBL,
        stel=STEL,
        sbel=SBEL,
        vbel=VBEL,
        vtel=VTEL,
        gnav=GNAV,
        time=0.0,
        trcond=0,
        trcvel=TRCVEL,
    )
    vehicle, guidance, ctx = _ready()
    guidance.execute(vehicle, ctx)
    store = vehicle.store
    np.testing.assert_allclose(store.get("alcomx"), want["alcomx"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ancomx"), want["ancomx"], rtol=RTOL, atol=ATOL)


def test_aa_clipped_to_gmax():
    unbounded = _cpp_term(
        thtpb=THTPB,
        psipb=PSIPB,
        sigdpy=SIGDPY,
        sigdpz=SIGDPZ,
        fspb=FSPB,
        gmax=1.0e9,
        tbl=TBL,
        stel=STEL,
        sbel=SBEL,
        vbel=VBEL,
        vtel=VTEL,
        gnav=GNAV,
        time=0.0,
        trcond=0,
        trcvel=TRCVEL,
    )
    unbounded_aa = sqrt(unbounded["all"] ** 2 + unbounded["ann"] ** 2)
    assert unbounded_aa > GMAX
    vehicle, guidance, ctx = _ready()
    guidance.execute(vehicle, ctx)
    store = vehicle.store
    all_ = store.get("all")
    ann = store.get("ann")
    np.testing.assert_allclose(all_, unbounded["all"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(ann, unbounded["ann"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(sqrt(all_ ** 2 + ann ** 2), unbounded_aa, rtol=RTOL, atol=ATOL)
    clipped = sqrt(store.get("alcomx") ** 2 + store.get("ancomx") ** 2)
    np.testing.assert_allclose(clipped, GMAX, rtol=RTOL, atol=ATOL)


def test_trcond_1_when_time_gt_3_and_dcvel_lt_trcvel():
    vehicle, guidance, ctx = _ready(time=3.1, trcvel=1.0e6)
    want = _cpp_term(
        thtpb=THTPB,
        psipb=PSIPB,
        sigdpy=SIGDPY,
        sigdpz=SIGDPZ,
        fspb=FSPB,
        gmax=GMAX,
        tbl=TBL,
        stel=STEL,
        sbel=SBEL,
        vbel=VBEL,
        vtel=VTEL,
        gnav=GNAV,
        time=3.1,
        trcond=0,
        trcvel=1.0e6,
    )
    assert want["dcvel"] < 1.0e6
    assert want["trcond"] == 1
    guidance.execute(vehicle, ctx)
    assert vehicle.store.get("trcond") == 1


def test_trcond_stays_0_when_time_le_3():
    vehicle, guidance, ctx = _ready(time=3.0, trcvel=1.0e6)
    guidance.execute(vehicle, ctx)
    assert vehicle.store.get("trcond") == 0


def test_trcond_stays_0_when_dcvel_ge_trcvel():
    vehicle, guidance, ctx = _ready(time=3.1, trcvel=TRCVEL)
    want = _cpp_term(
        thtpb=THTPB,
        psipb=PSIPB,
        sigdpy=SIGDPY,
        sigdpz=SIGDPZ,
        fspb=FSPB,
        gmax=GMAX,
        tbl=TBL,
        stel=STEL,
        sbel=SBEL,
        vbel=VBEL,
        vtel=VTEL,
        gnav=GNAV,
        time=3.1,
        trcond=0,
        trcvel=TRCVEL,
    )
    assert want["dcvel"] >= TRCVEL
    guidance.execute(vehicle, ctx)
    assert vehicle.store.get("trcond") == 0

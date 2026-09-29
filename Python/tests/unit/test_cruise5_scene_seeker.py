"""CRUISE5 S1/S1LOS/S1EPCH scene-matching imaging seeker (Task 73)."""

import numpy as np

from cadac.constants import DEG
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round3.cruise5.seeker import Cruise5Seeker

RTOL = 1e-12
ATOL = 1e-12
INT_STEP = 0.05

# Geometry: vehicle at origin-ish L, waypoint 500 m ahead, alt match → DWB=500
SBEL0 = np.array([0.0, 0.0, -1000.0])
SWEL = np.array([500.0, 0.0, -1000.0])
# Perfect INS: SBWLC = vehicle wrt waypoint = SBEL - SWEL
SBWLC0 = SBEL0 - SWEL
TBL0 = np.eye(3)
TBLC0 = np.eye(3)

DTIMMP = 0.5
DTIMCR = 0.5
RACQ = 1000.0
RMIN = 100.0
NFIXM = 5
NMAP = 1


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(sim_time=0.0):
    return SimContext(sim_time, INT_STEP, 0.0, 0.0, [], 0)


def _ready(**overrides):
    """Plant Fortran S1 inputs; zero noise; wide FOV/FOR."""
    vehicle = _Vehicle()
    seeker = Cruise5Seeker()
    seeker.define(vehicle)
    store = vehicle.store
    for name, value, ftype, role, module in (
        ("SBEL", SBEL0.copy(), "vec", "state", "newton"),
        ("TBL", TBL0.copy(), "mat", "out", "newton"),
        ("TBLC", TBLC0.copy(), "mat", "out", "ins"),
        ("SBWLC", SBWLC0.copy(), "vec", "out", "ins"),
    ):
        if name not in store:
            store.define(Field(name, value, ftype, role, module))
        else:
            store.set(name, value)

    store.set("mseeker", 1)
    store.set("nmap", NMAP)
    store.set("nfixm", NFIXM)
    store.set("racq", RACQ)
    store.set("rmin", RMIN)
    store.set("SWEL", SWEL.copy())
    store.set("SWRWL", np.zeros(3))
    store.set("randpb", 0.0)
    store.set("randtb", 0.0)
    store.set("randpc", 0.0)
    store.set("randtc", 0.0)
    store.set("randdc", 0.0)
    store.set("dtimmp", DTIMMP)
    store.set("dtimcr", DTIMCR)
    store.set("fovyaw", 0.5)
    store.set("fovpit", 0.5)
    store.set("foryaw", 1.0)
    store.set("forpit", 1.0)
    for name, value in overrides.items():
        store.set(name, value)
    seeker.initialize(vehicle, _ctx(0.0))
    return vehicle, seeker


def test_scene_seeker_mseeker_1_to_2_sets_epochs():
    """Within RACQ + new NMAP → NFIX=1, MSEEK=2, S1EPCH epochs."""
    veh, seeker = _ready()
    seeker.execute(veh, _ctx(0.0))
    store = veh.store
    assert store.get("mseeker") == 2
    assert store.get("nfix") == 1
    assert abs(store.get("epchn3") - DTIMMP) < 1e-12
    assert abs(store.get("epchn4") - (DTIMMP + DTIMCR)) < 1e-12
    assert abs(store.get("dtimfx") - (DTIMMP + DTIMCR)) < 1e-12
    np.testing.assert_allclose(store.get("SWREL"), SWEL, rtol=RTOL, atol=ATOL)
    assert abs(store.get("dwb") - 500.0) < 1e-12


def test_scene_seeker_advances_2_to_3_to_4_and_increments_nfix():
    """Drive epochs: T>=EPCHN3 → MSEEK=3 (map store); T>=EPCHN4 → MSEEK=4, NFIX++."""
    veh, seeker = _ready()
    # t=0: enable → imaging
    seeker.execute(veh, _ctx(0.0))
    assert veh.store.get("mseeker") == 2
    assert veh.store.get("nfix") == 1
    epchn3 = veh.store.get("epchn3")
    epchn4 = veh.store.get("epchn4")
    assert abs(epchn3 - DTIMMP) < 1e-12
    assert abs(epchn4 - (DTIMMP + DTIMCR)) < 1e-12

    # t just before scene epoch: still 2
    seeker.execute(veh, _ctx(epchn3 - 0.01))
    assert veh.store.get("mseeker") == 2

    # scene-taking epoch
    seeker.execute(veh, _ctx(epchn3))
    assert veh.store.get("mseeker") == 3
    assert veh.store.get("nfix") == 1

    # correlation/update epoch → update sent, nfix increments
    seeker.execute(veh, _ctx(epchn4))
    assert veh.store.get("mseeker") == 4
    assert veh.store.get("nfix") == 2
    swalc = veh.store.get("SWALC")
    assert swalc is not None
    assert swalc.shape == (3,)
    # zero noise + perfect INS → update correction near zero
    np.testing.assert_allclose(swalc, np.zeros(3), atol=1e-9)


def test_scene_seeker_second_fix_uses_dtimcr_only():
    """NFIX>1: EPCHN3=T, DTIMFX=DTIMCR; same step also hits scene epoch → MSEEK=3."""
    veh, seeker = _ready()
    seeker.execute(veh, _ctx(0.0))
    epchn4 = veh.store.get("epchn4")
    seeker.execute(veh, _ctx(epchn4))
    assert veh.store.get("mseeker") == 4
    assert veh.store.get("nfix") == 2

    t2 = epchn4 + INT_STEP
    seeker.execute(veh, _ctx(t2))
    # S1EPCH sets EPCHN3=T for NFIX>1, so T.GE.EPCHN3 fires same step → MSEEK=3
    assert veh.store.get("mseeker") == 3
    assert veh.store.get("nfix") == 2
    assert abs(veh.store.get("epchn3") - t2) < 1e-12
    assert abs(veh.store.get("dtimfx") - DTIMCR) < 1e-12
    assert abs(veh.store.get("epchn4") - (t2 + DTIMCR)) < 1e-12


def test_los_path_untouched_when_nmap_zero():
    """nmap=0 keeps Task 2 LOS acquire (mseeker 1→3 via TARGET3), not scene 1→2."""
    from cadac.constants import RAD, REARTH
    from cadac.kernel.combus import Packet
    from cadac.math.earth import cadine, cadtei, cadtge
    from cadac.math.frames import mat2tr

    lonx, latx, alt = 14.7, 35.4, 7000.0
    range_m = 500.0
    dlat = (range_m / REARTH) * DEG
    tgt_latx = latx + dlat
    tei = cadtei(0.0)
    tge = cadtge(lonx * RAD, latx * RAD)
    tig = tei.T @ tge.T
    sbii = cadine(lonx * RAD, latx * RAD, alt, 0.0)
    vbeg = mat2tr(90.0 * RAD, 0.0).T @ np.array([200.0, 0.0, 0.0])
    _, tgt_sbii, tgt_vbeg = (
        None,
        cadine(lonx * RAD, tgt_latx * RAD, 0.0, 0.0),
        np.zeros(3),
    )

    vehicle = _Vehicle()
    seeker = Cruise5Seeker()
    seeker.define(vehicle)
    store = vehicle.store
    for name, value, ftype, role, module in (
        ("tig", tig, "mat", "init/out", "newton"),
        ("vbeg", vbeg, "vec", "state", "newton"),
        ("sbii", sbii, "vec", "state", "newton"),
        ("TBG", np.eye(3), "mat", "out", "control"),
        ("lonx", lonx, "real", "init/diag", "newton"),
        ("latx", latx, "real", "init/diag", "newton"),
    ):
        store.define(Field(name, value, ftype, role, module))
    store.set("mseeker", 1)
    store.set("nmap", 0)
    store.set("acq_range", 1000.0)
    seeker.initialize(vehicle, _ctx(0.0))
    combus = [
        Packet(
            name="t1",
            type="TARGET3",
            status=1,
            vars={
                "lonx": lonx,
                "latx": tgt_latx,
                "alt": 0.0,
                "psivgx": 0.0,
                "thtvgx": 0.0,
                "dvbe": 0.0,
                "vbeg": tgt_vbeg,
                "sbii": tgt_sbii,
            },
        ),
    ]
    seeker.execute(vehicle, SimContext(0.0, INT_STEP, 0.0, 0.0, combus, 0))
    assert store.get("mseeker") == 3
    assert store.get("acquisition") == 1

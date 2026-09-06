import numpy as np
import pytest

from cadac.constants import RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.wgs84 import cad_in_geo84
from cadac.vehicles.rocket6.startrack import (
    STAR_CATALOG,
    Rocket6Startrack,
    star_init,
    star_triad,
)

RTOL = 1e-12
ATOL = 1e-14

SIRIUS = (-0.179457, 0.947482, -0.264715)
LONX = -120.49
LATX = 34.68
ALT_PAD = 100.0
TIME = 0.0
STARTRACK_ALT = 30000.0
STAR_ACQTIME = 20.0
STAR_STEP = 10.0
STAR_EL_MIN = 1.0


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(int_step=0.001, sim_time=TIME):
    return SimContext(
        sim_time=sim_time,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _plant_time_alt(store, alt):
    for name, value, ftype, role, module in (
        ("time", TIME, "real", "exec", "kinematics"),
        ("alt", alt, "real", "out", "newton"),
    ):
        if name not in store.names():
            store.define(Field(name, value, ftype, role, module))
        store.set(name, value)


def _ready(mstar, alt=0.0, star_acqtime=STAR_ACQTIME):
    vehicle = _Vehicle()
    startrack = Rocket6Startrack()
    startrack.define(vehicle)
    _plant_time_alt(vehicle.store, alt)
    vehicle.store.set("mstar", mstar)
    vehicle.store.set("startrack_alt", STARTRACK_ALT)
    vehicle.store.set("star_acqtime", star_acqtime)
    vehicle.store.set("star_step", STAR_STEP)
    vehicle.store.set("star_el_min", STAR_EL_MIN)
    startrack.initialize(vehicle, _ctx())
    return startrack, vehicle


def test_name_is_startrack():
    # Break: class name token not "startrack" (module bind).
    assert Rocket6Startrack().name == "startrack"


def test_catalog_row0_sirius():
    # Break: catalog not copied from C++ star_init; row 0 is Sirius J2000.
    np.testing.assert_allclose(STAR_CATALOG[0], SIRIUS, rtol=RTOL, atol=ATOL)
    star_data, star_names = star_init()
    np.testing.assert_allclose(star_data[0], SIRIUS, rtol=RTOL, atol=ATOL)
    assert star_names[0] == "Sirius"
    assert STAR_CATALOG.shape == (25, 3)
    assert star_data.shape == (25, 3)


def test_mstar_zero_no_raise():
    # Break: mstar=0 raises or writes tracking state.
    vehicle = _Vehicle()
    startrack = Rocket6Startrack()
    startrack.define(vehicle)
    startrack.initialize(vehicle, _ctx())
    startrack.execute(vehicle, _ctx())
    assert vehicle.store.get("mstar") == 0
    assert vehicle.store.get("star_acq") == 0
    np.testing.assert_array_equal(vehicle.store.get("URIC"), np.zeros(3))


def test_mstar_one_same_call_sets_two_when_above_alt():
    # Break: elif after init leaves wait-clock unrun, or alt gate skipped so mstar stays 1,
    # or star_acqtime==0 fall-through sets mstar=3 this call.
    startrack, vehicle = _ready(1, alt=STARTRACK_ALT + 1000.0)
    store = vehicle.store
    startrack.execute(vehicle, _ctx())
    assert store.get("mstar") == 2
    assert store.get("star_acq") == 1
    assert store.get("starfix_epoch") == TIME


def test_mstar_one_at_alt_zero_stays_one():
    # Break: C++ alt>startrack_alt gate dropped so mstar becomes 2 at the pad.
    startrack, vehicle = _ready(1, alt=0.0)
    startrack.execute(vehicle, _ctx())
    assert vehicle.store.get("mstar") == 1
    assert vehicle.store.get("star_acq") == 1
    assert vehicle.store.get("starfix_epoch") == TIME


def test_mstar_four_raises():
    # Break: mstar not in {0,1,2,3} is accepted.
    startrack, vehicle = _ready(4)
    with pytest.raises(ValueError, match="mstar"):
        startrack.execute(vehicle, _ctx())


def test_triad_volume_in_unit_interval():
    # Break: star_triad not ported or max-volume pick yields 0 / >1.
    sbii = cad_in_geo84(LONX * RAD, LATX * RAD, ALT_PAD, TIME)
    star_data, _names = star_init()
    usii_triad, volume = star_triad(star_data, STAR_EL_MIN, sbii)
    assert 0.0 < volume <= 1.0
    assert usii_triad.shape == (3, 4)
    slots = [int(usii_triad[i, 3]) for i in range(3)]
    assert all(1 <= slot <= 25 for slot in slots)
    assert len(set(slots)) == 3
    assert np.all(np.isfinite(usii_triad[:, :3]))

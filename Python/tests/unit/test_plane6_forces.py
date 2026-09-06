from types import SimpleNamespace

import numpy as np

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.plane6.forces import Plane6Forces

RTOL = 1e-12
ATOL = 1e-14

# F-16 geometry (init_aerodynamics) and a frozen aero/thrust point
PDYNMC = 17000.0
REFA = 27.87
REFB = 9.14
REFC = 3.45
THRUST = 25000.0
CXT = -0.04
CYT = 0.02
CZT = -0.35
CLT = 0.003
CMT = -0.06
CNT = 0.0015

DEFINED = ("FAPB", "FMB")
EXTERNALS = (
    "pdynmc",
    "thrust",
    "refa",
    "refb",
    "refc",
    "cxt",
    "cyt",
    "czt",
    "clt",
    "cmt",
    "cnt",
    "FSPB",
    "vmass",
    "time",
)


def _ctx(int_step=0.001):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _expected_fapb_fmb(
    pdynmc, refa, refb, refc, cxt, cyt, czt, clt, cmt, cnt, thrust
):
    fapb = np.array(
        [
            pdynmc * refa * cxt + thrust,
            pdynmc * refa * cyt,
            pdynmc * refa * czt,
        ],
        dtype=float,
    )
    fmb = np.array(
        [
            pdynmc * refa * refb * clt,
            pdynmc * refa * refc * cmt,
            pdynmc * refa * refb * cnt,
        ],
        dtype=float,
    )
    return fapb, fmb


def _inputs(
    store,
    *,
    pdynmc=PDYNMC,
    refa=REFA,
    refb=REFB,
    refc=REFC,
    cxt=CXT,
    cyt=CYT,
    czt=CZT,
    clt=CLT,
    cmt=CMT,
    cnt=CNT,
    thrust=THRUST,
):
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("thrust", thrust, "real", "out", "propulsion"))
    store.define(Field("refa", refa, "real", "init", "aerodynamics"))
    store.define(Field("refb", refb, "real", "init", "aerodynamics"))
    store.define(Field("refc", refc, "real", "init", "aerodynamics"))
    store.define(Field("cxt", cxt, "real", "out", "aerodynamics"))
    store.define(Field("cyt", cyt, "real", "out", "aerodynamics"))
    store.define(Field("czt", czt, "real", "out", "aerodynamics"))
    store.define(Field("clt", clt, "real", "out", "aerodynamics"))
    store.define(Field("cmt", cmt, "real", "out", "aerodynamics"))
    store.define(Field("cnt", cnt, "real", "out", "aerodynamics"))


def _ready(**kwargs):
    vehicle = SimpleNamespace(store=StateStore())
    forces = Plane6Forces()
    forces.define(vehicle)
    _inputs(vehicle.store, **kwargs)
    forces.initialize(vehicle, _ctx())
    return vehicle, forces


def test_name_is_forces():
    assert Plane6Forces().name == "forces"


def test_define_registers_fapb_fmb_only():
    vehicle = SimpleNamespace(store=StateStore())
    Plane6Forces().define(vehicle)
    store = vehicle.store
    zeros = np.zeros(3)
    assert list(store.names()) == list(DEFINED)
    for name in DEFINED:
        field = store.field(name)
        np.testing.assert_array_equal(store.get(name), zeros)
        assert store.get(name).shape == (3,)
        assert field.type == "vec"
        assert field.role == "out"
        assert field.module == "forces"
        assert field.outputs == ()
    for name in EXTERNALS:
        assert name not in store.names()


def test_initialize_is_pass():
    vehicle, _forces = _ready()
    store = vehicle.store
    np.testing.assert_array_equal(store.get("FAPB"), np.zeros(3))
    np.testing.assert_array_equal(store.get("FMB"), np.zeros(3))


def test_fapb_fmb_match_cadac_at_frozen_point():
    vehicle, forces = _ready()
    store = vehicle.store
    want_fapb, want_fmb = _expected_fapb_fmb(
        PDYNMC, REFA, REFB, REFC, CXT, CYT, CZT, CLT, CMT, CNT, THRUST
    )
    assert REFB != REFC
    assert THRUST != 0.0
    assert np.all(want_fapb != 0.0)
    assert np.all(want_fmb != 0.0)

    forces.execute(vehicle, _ctx())

    np.testing.assert_allclose(store.get("FAPB"), want_fapb, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FMB"), want_fmb, rtol=RTOL, atol=ATOL)
    assert store.get("FAPB").shape == (3,)
    assert store.get("FMB").shape == (3,)


def test_does_not_write_fspb_or_require_vmass_time():
    vehicle, forces = _ready()
    store = vehicle.store
    assert "FSPB" not in store.names()
    assert "vmass" not in store.names()
    assert "time" not in store.names()
    sentinel = np.array([9.0, 8.0, 7.0])
    store.define(Field("FSPB", sentinel, "vec", "out", "newton"))
    forces.execute(vehicle, _ctx())
    np.testing.assert_array_equal(store.get("FSPB"), sentinel)
    assert "vmass" not in store.names()
    assert "time" not in store.names()

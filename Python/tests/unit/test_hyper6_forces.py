from types import SimpleNamespace

import numpy as np

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round6.hyper6.forces import Hyper6Forces

RTOL = 1e-12
ATOL = 1e-14

# GHAME geometry (Hyper::init_aerodynamics) and a frozen aero/thrust point
PDYNMC = 200000.0
REFA = 557.42
REFB = 24.38
REFC = 22.86
THRUST = 1.0e6
CX = -0.04
CY = 0.02
CZ = -0.35
CLL = 0.003
CLM = -0.06
CLN = 0.0015

# C++ Hyper::forces with FARCS=FMRCS=0
WANT_FAPB = np.array(
    [
        -3459359.999999999,
        2229679.9999999995,
        -39019399.99999999,
    ],
    dtype=float,
)
WANT_FMB = np.array(
    [
        8153939.759999999,
        -152911454.39999998,
        4076969.8799999994,
    ],
    dtype=float,
)

DEFINED = ("FAPB", "FMB")
EXTERNALS = (
    "pdynmc",
    "thrust",
    "refa",
    "refb",
    "refc",
    "cx",
    "cy",
    "cz",
    "cll",
    "clm",
    "cln",
    "FARCS",
    "FMRCS",
    "FSPB",
    "vmass",
    "time",
)


def _ctx(int_step=0.01):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _inputs(
    store,
    *,
    pdynmc=PDYNMC,
    refa=REFA,
    refb=REFB,
    refc=REFC,
    cx=CX,
    cy=CY,
    cz=CZ,
    cll=CLL,
    clm=CLM,
    cln=CLN,
    thrust=THRUST,
):
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("thrust", thrust, "real", "out", "propulsion"))
    store.define(Field("refa", refa, "real", "init", "aerodynamics"))
    store.define(Field("refb", refb, "real", "init", "aerodynamics"))
    store.define(Field("refc", refc, "real", "init", "aerodynamics"))
    store.define(Field("cx", cx, "real", "out", "aerodynamics"))
    store.define(Field("cy", cy, "real", "out", "aerodynamics"))
    store.define(Field("cz", cz, "real", "out", "aerodynamics"))
    store.define(Field("cll", cll, "real", "out", "aerodynamics"))
    store.define(Field("clm", clm, "real", "out", "aerodynamics"))
    store.define(Field("cln", cln, "real", "out", "aerodynamics"))


def _ready(**kwargs):
    vehicle = SimpleNamespace(store=StateStore())
    forces = Hyper6Forces()
    forces.define(vehicle)
    _inputs(vehicle.store, **kwargs)
    forces.initialize(vehicle, _ctx())
    return vehicle, forces


def test_name_is_forces():
    assert Hyper6Forces().name == "forces"


def test_define_registers_fapb_fmb_only():
    vehicle = SimpleNamespace(store=StateStore())
    Hyper6Forces().define(vehicle)
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
    assert REFB != REFC
    assert THRUST != 0.0
    assert np.all(WANT_FAPB != 0.0)
    assert np.all(WANT_FMB != 0.0)
    assert "FARCS" not in store.names()
    assert "FMRCS" not in store.names()

    forces.execute(vehicle, _ctx())

    np.testing.assert_allclose(store.get("FAPB"), WANT_FAPB, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FMB"), WANT_FMB, rtol=RTOL, atol=ATOL)
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


def test_absent_farcs_fmrcs_treated_as_zero():
    vehicle, forces = _ready()
    store = vehicle.store
    forces.execute(vehicle, _ctx())
    np.testing.assert_allclose(store.get("FAPB"), WANT_FAPB, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FMB"), WANT_FMB, rtol=RTOL, atol=ATOL)


def test_present_farcs_fmrcs_are_added():
    vehicle, forces = _ready()
    store = vehicle.store
    farcs = np.array([100.0, -20.0, 3.0])
    fmrcs = np.array([-4.0, 5.0, -6.0])
    store.define(Field("FARCS", farcs, "vec", "out", "rcs"))
    store.define(Field("FMRCS", fmrcs, "vec", "out", "rcs"))
    forces.execute(vehicle, _ctx())
    np.testing.assert_allclose(store.get("FAPB"), WANT_FAPB + farcs, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FMB"), WANT_FMB + fmrcs, rtol=RTOL, atol=ATOL)


def test_terminate_exists_and_is_pass():
    vehicle, forces = _ready()
    store = vehicle.store
    forces.terminate(vehicle, _ctx())
    np.testing.assert_array_equal(store.get("FAPB"), np.zeros(3))
    np.testing.assert_array_equal(store.get("FMB"), np.zeros(3))
    assert "FSPB" not in store.names()

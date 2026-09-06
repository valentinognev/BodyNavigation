from types import SimpleNamespace

import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.rocket6.forces import Rocket6Forces

RTOL = 1e-12
ATOL = 1e-14

# insertion SLV geometry (input.asc) and a frozen aero/thrust point
PDYNMC = 5000.0
REFA = 3.243
REFD = 2.032
THRUST = 1.0e6
CX = -0.12
CY = 0.03
CZ = -0.25
CLL = 0.004
CLM = -0.08
CLN = 0.002
MPROP = 3

FPB = np.array([100.0, -20.0, 30.0], dtype=float)
FMPB = np.array([4.0, -5.0, 6.0], dtype=float)
FARCS = np.array([10.0, -8.0, 7.0], dtype=float)
FMRCS = np.array([-1.0, 2.0, -3.0], dtype=float)

DEFINED = ("FAPB", "FMB")
EXTERNALS = (
    "pdynmc",
    "mprop",
    "thrust",
    "mrcs_moment",
    "mrcs_force",
    "refa",
    "refd",
    "cx",
    "cy",
    "cz",
    "cll",
    "clm",
    "cln",
    "mtvc",
    "FPB",
    "FMPB",
    "FARCS",
    "FMRCS",
    "FSPB",
    "refb",
    "refc",
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


def _aero_force(cx=CX, cy=CY, cz=CZ, pdynmc=PDYNMC, refa=REFA):
    return np.array(
        [pdynmc * refa * cx, pdynmc * refa * cy, pdynmc * refa * cz],
        dtype=float,
    )


def _aero_moment(cll=CLL, clm=CLM, cln=CLN, pdynmc=PDYNMC, refa=REFA, refd=REFD):
    return np.array(
        [
            pdynmc * refa * refd * cll,
            pdynmc * refa * refd * clm,
            pdynmc * refa * refd * cln,
        ],
        dtype=float,
    )


def _inputs(
    store,
    *,
    pdynmc=PDYNMC,
    mprop=MPROP,
    thrust=THRUST,
    mrcs_moment=0,
    mrcs_force=0,
    refa=REFA,
    refd=REFD,
    cx=CX,
    cy=CY,
    cz=CZ,
    cll=CLL,
    clm=CLM,
    cln=CLN,
    mtvc=0,
):
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("mprop", mprop, "int", "data", "propulsion"))
    store.define(Field("thrust", thrust, "real", "out", "propulsion"))
    store.define(Field("mrcs_moment", mrcs_moment, "int", "data", "rcs"))
    store.define(Field("mrcs_force", mrcs_force, "int", "data", "rcs"))
    store.define(Field("refa", refa, "real", "init", "aerodynamics"))
    store.define(Field("refd", refd, "real", "init", "aerodynamics"))
    store.define(Field("cx", cx, "real", "out", "aerodynamics"))
    store.define(Field("cy", cy, "real", "out", "aerodynamics"))
    store.define(Field("cz", cz, "real", "out", "aerodynamics"))
    store.define(Field("cll", cll, "real", "out", "aerodynamics"))
    store.define(Field("clm", clm, "real", "out", "aerodynamics"))
    store.define(Field("cln", cln, "real", "out", "aerodynamics"))
    store.define(Field("mtvc", mtvc, "int", "data", "tvc"))


def _ready(**kwargs):
    vehicle = SimpleNamespace(store=StateStore())
    forces = Rocket6Forces()
    forces.define(vehicle)
    _inputs(vehicle.store, **kwargs)
    forces.initialize(vehicle, _ctx())
    return vehicle, forces


def test_name_is_forces():
    assert Rocket6Forces().name == "forces"


def test_define_registers_fapb_fmb_only():
    vehicle = SimpleNamespace(store=StateStore())
    Rocket6Forces().define(vehicle)
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


def test_mtvc0_mprop3_adds_thrust_to_axial_aero():
    vehicle, forces = _ready(mtvc=0, mprop=3)
    store = vehicle.store
    assert "FPB" not in store.names()
    assert "FMRCS" not in store.names()
    assert THRUST != 0.0
    want_fapb = _aero_force()
    want_fapb[0] = want_fapb[0] + THRUST
    want_fmb = _aero_moment()

    forces.execute(vehicle, _ctx())

    np.testing.assert_allclose(
        store.get("FAPB")[0], PDYNMC * REFA * CX + THRUST, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(store.get("FAPB"), want_fapb, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        store.get("FMB")[1], PDYNMC * REFA * REFD * CLM, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(store.get("FMB"), want_fmb, rtol=RTOL, atol=ATOL)
    assert store.get("FAPB").shape == (3,)
    assert store.get("FMB").shape == (3,)


def test_fmb_uses_refd_not_ghame_refb_refc():
    vehicle, forces = _ready()
    store = vehicle.store
    assert "refb" not in store.names()
    assert "refc" not in store.names()
    assert REFD != REFA

    forces.execute(vehicle, _ctx())

    np.testing.assert_allclose(
        store.get("FMB"), _aero_moment(), rtol=RTOL, atol=ATOL
    )
    wrong_ghame = np.array(
        [
            PDYNMC * REFA * REFA * CLL,
            PDYNMC * REFA * (REFA + 1.0) * CLM,
            PDYNMC * REFA * REFA * CLN,
        ],
        dtype=float,
    )
    assert not np.allclose(store.get("FMB"), wrong_ghame, rtol=RTOL, atol=ATOL)


def test_mtvc2_adds_fpb_and_skips_plain_thrust():
    vehicle, forces = _ready(mtvc=2, mprop=3)
    store = vehicle.store
    store.define(Field("FPB", FPB, "vec", "out", "tvc"))
    store.define(Field("FMPB", FMPB, "vec", "out", "tvc"))

    forces.execute(vehicle, _ctx())

    want_fapb = _aero_force() + FPB
    want_fmb = _aero_moment() + FMPB
    np.testing.assert_allclose(store.get("FAPB"), want_fapb, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FMB"), want_fmb, rtol=RTOL, atol=ATOL)
    with_thrust = _aero_force()
    with_thrust[0] = with_thrust[0] + THRUST
    assert not np.allclose(store.get("FAPB"), with_thrust, rtol=RTOL, atol=ATOL)
    assert not np.allclose(store.get("FAPB"), with_thrust + FPB, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("mtvc", [1, 3])
def test_mtvc_1_and_3_add_fpb_fmpb(mtvc):
    vehicle, forces = _ready(mtvc=mtvc, mprop=3)
    store = vehicle.store
    store.define(Field("FPB", FPB, "vec", "out", "tvc"))
    store.define(Field("FMPB", FMPB, "vec", "out", "tvc"))
    forces.execute(vehicle, _ctx())
    np.testing.assert_allclose(store.get("FAPB"), _aero_force() + FPB, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FMB"), _aero_moment() + FMPB, rtol=RTOL, atol=ATOL)


def test_does_not_write_fspb():
    vehicle, forces = _ready()
    store = vehicle.store
    assert "FSPB" not in store.names()
    sentinel = np.array([9.0, 8.0, 7.0])
    store.define(Field("FSPB", sentinel, "vec", "out", "newton"))
    forces.execute(vehicle, _ctx())
    np.testing.assert_array_equal(store.get("FSPB"), sentinel)
    assert store.field("FSPB").module == "newton"


def test_missing_farcs_fmrcs_fpb_treated_as_zero():
    vehicle, forces = _ready(mtvc=2, mprop=3, mrcs_moment=21, mrcs_force=1)
    store = vehicle.store
    assert "FARCS" not in store.names()
    assert "FMRCS" not in store.names()
    assert "FPB" not in store.names()
    assert "FMPB" not in store.names()

    forces.execute(vehicle, _ctx())

    np.testing.assert_allclose(store.get("FAPB"), _aero_force(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FMB"), _aero_moment(), rtol=RTOL, atol=ATOL)


def test_mprop0_mtvc0_does_not_add_thrust():
    vehicle, forces = _ready(mtvc=0, mprop=0, thrust=THRUST)
    forces.execute(vehicle, _ctx())
    np.testing.assert_allclose(
        vehicle.store.get("FAPB"), _aero_force(), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("mrcs_moment", [1, 21, 23])
def test_fmrcs_added_when_mrcs_moment_in_range(mrcs_moment):
    vehicle, forces = _ready(mtvc=0, mprop=0, mrcs_moment=mrcs_moment)
    store = vehicle.store
    store.define(Field("FMRCS", FMRCS, "vec", "out", "rcs"))
    forces.execute(vehicle, _ctx())
    np.testing.assert_allclose(
        store.get("FMB"), _aero_moment() + FMRCS, rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("mrcs_moment", [0, 24])
def test_fmrcs_not_added_outside_range(mrcs_moment):
    vehicle, forces = _ready(mtvc=0, mprop=0, mrcs_moment=mrcs_moment)
    store = vehicle.store
    store.define(Field("FMRCS", FMRCS, "vec", "out", "rcs"))
    forces.execute(vehicle, _ctx())
    np.testing.assert_allclose(store.get("FMB"), _aero_moment(), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("mrcs_force", [1, 2])
def test_farcs_added_when_mrcs_force_enabled(mrcs_force):
    vehicle, forces = _ready(mtvc=0, mprop=0, mrcs_force=mrcs_force)
    store = vehicle.store
    store.define(Field("FARCS", FARCS, "vec", "out", "rcs"))
    forces.execute(vehicle, _ctx())
    np.testing.assert_allclose(
        store.get("FAPB"), _aero_force() + FARCS, rtol=RTOL, atol=ATOL
    )


def test_farcs_not_added_when_mrcs_force_zero():
    vehicle, forces = _ready(mtvc=0, mprop=0, mrcs_force=0)
    store = vehicle.store
    store.define(Field("FARCS", FARCS, "vec", "out", "rcs"))
    forces.execute(vehicle, _ctx())
    np.testing.assert_allclose(store.get("FAPB"), _aero_force(), rtol=RTOL, atol=ATOL)


def test_terminate_exists_and_is_pass():
    vehicle, forces = _ready()
    store = vehicle.store
    forces.terminate(vehicle, _ctx())
    np.testing.assert_array_equal(store.get("FAPB"), np.zeros(3))
    np.testing.assert_array_equal(store.get("FMB"), np.zeros(3))
    assert "FSPB" not in store.names()

from pathlib import Path

import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.sam6.forces import Sam6Forces

RTOL = 1e-12
ATOL = 1e-14

PDYNMC = 156.0
REFA = 0.0491
REFL = 0.25
CA = 0.4
THRUST = 1000.0

DEFINED = ("FAPB", "FMB")
NOT_DEFINED = (
    "pdynmc",
    "thrust",
    "refa",
    "refl",
    "ca",
    "cy",
    "cn",
    "cll",
    "clm",
    "cln",
    "mtvc",
    "FARCS",
    "FMRCS",
    "FPB",
    "FMPB",
    "FSPB",
    "mass",
    "time",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(int_step=0.001):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _plant(
    store,
    *,
    pdynmc=PDYNMC,
    refa=REFA,
    refl=REFL,
    ca=CA,
    cy=0.0,
    cn=0.0,
    cll=0.0,
    clm=0.0,
    cln=0.0,
    thrust=THRUST,
    mtvc=0,
):
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("refa", refa, "real", "init", "aerodynamics"))
    store.define(Field("refl", refl, "real", "init", "aerodynamics"))
    store.define(Field("ca", ca, "real", "out", "aerodynamics"))
    store.define(Field("cy", cy, "real", "out", "aerodynamics"))
    store.define(Field("cn", cn, "real", "out", "aerodynamics"))
    store.define(Field("cll", cll, "real", "out", "aerodynamics"))
    store.define(Field("clm", clm, "real", "out", "aerodynamics"))
    store.define(Field("cln", cln, "real", "out", "aerodynamics"))
    store.define(Field("thrust", thrust, "real", "out", "propulsion"))
    if mtvc is not None:
        store.define(Field("mtvc", mtvc, "int", "data", "tvc"))


def _ready(**kwargs):
    vehicle = _Vehicle()
    forces = Sam6Forces()
    forces.define(vehicle)
    forces.initialize(vehicle, _ctx())
    _plant(vehicle.store, **kwargs)
    return vehicle, forces


def _cpp_fapb(*, pdynmc, refa, ca, cy, cn, thrust, farcs=(0.0, 0.0, 0.0)):
    return np.array(
        [
            -pdynmc * refa * ca + thrust + farcs[0],
            pdynmc * refa * cy + farcs[1],
            -pdynmc * refa * cn + farcs[2],
        ],
        dtype=float,
    )


def _cpp_fmb(*, pdynmc, refa, refl, cll, clm, cln, fmrcs=(0.0, 0.0, 0.0)):
    return np.array(
        [
            pdynmc * refa * refl * cll + fmrcs[0],
            pdynmc * refa * refl * clm + fmrcs[1],
            pdynmc * refa * refl * cln + fmrcs[2],
        ],
        dtype=float,
    )


def test_name_is_forces():
    assert Sam6Forces().name == "forces"


def test_define_registers_fapb_fmb_only():
    vehicle = _Vehicle()
    Sam6Forces().define(vehicle)
    store = vehicle.store
    zeros = np.zeros(3)
    assert tuple(store.names()) == DEFINED
    for name in DEFINED:
        field = store.field(name)
        np.testing.assert_array_equal(store.get(name), zeros)
        assert store.get(name).shape == (3,)
        assert field.type == "vec"
        assert field.role == "out"
        assert field.module == "forces"
        assert field.outputs == ()
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_initialize_is_pass():
    vehicle = _Vehicle()
    forces = Sam6Forces()
    forces.define(vehicle)
    before = {name: np.array(vehicle.store.get(name), copy=True) for name in DEFINED}
    assert forces.initialize(vehicle, _ctx()) is None
    for name in DEFINED:
        np.testing.assert_allclose(
            vehicle.store.get(name), before[name], rtol=RTOL, atol=ATOL
        )


def test_fapb0_mtvc_zero_adds_thrust():
    vehicle, forces = _ready()
    store = vehicle.store
    assert "FARCS" not in store.names()
    assert "FMRCS" not in store.names()
    forces.execute(vehicle, _ctx())
    want = -PDYNMC * REFA * CA + THRUST
    np.testing.assert_allclose(store.get("FAPB")[0], want, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        store.get("FAPB")[0], -156 * 0.0491 * 0.4 + 1000, rtol=RTOL, atol=ATOL
    )


def test_mtvc_nonzero_raises():
    vehicle, forces = _ready(mtvc=1)
    store = vehicle.store
    store.define(Field("FPB", (9.0, 8.0, 7.0), "vec", "out", "tvc"))
    store.define(Field("FMPB", (6.0, 5.0, 4.0), "vec", "out", "tvc"))
    with pytest.raises(ValueError):
        forces.execute(vehicle, _ctx())
    np.testing.assert_array_equal(store.get("FAPB"), np.zeros(3))
    np.testing.assert_array_equal(store.get("FMB"), np.zeros(3))


def test_aero_sums_match_cpp_when_coeffs_nonzero():
    cy, cn, cll, clm, cln = 0.1, 0.2, 0.01, -0.03, 0.004
    vehicle, forces = _ready(cy=cy, cn=cn, cll=cll, clm=clm, cln=cln)
    forces.execute(vehicle, _ctx())
    store = vehicle.store
    want_fapb = _cpp_fapb(
        pdynmc=PDYNMC, refa=REFA, ca=CA, cy=cy, cn=cn, thrust=THRUST
    )
    want_fmb = _cpp_fmb(
        pdynmc=PDYNMC, refa=REFA, refl=REFL, cll=cll, clm=clm, cln=cln
    )
    np.testing.assert_allclose(store.get("FAPB"), want_fapb, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FMB"), want_fmb, rtol=RTOL, atol=ATOL)
    assert store.get("FAPB").shape == (3,)
    assert store.get("FMB").shape == (3,)


def test_zero_side_coeffs_leave_fapb12_and_fmb_zero():
    vehicle, forces = _ready()
    forces.execute(vehicle, _ctx())
    store = vehicle.store
    np.testing.assert_allclose(store.get("FAPB")[1], 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FAPB")[2], 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FMB"), np.zeros(3), rtol=RTOL, atol=ATOL)


def test_does_not_write_fspb_or_require_time():
    vehicle, forces = _ready()
    store = vehicle.store
    assert "FSPB" not in store.names()
    assert "time" not in store.names()
    sentinel = np.array([9.0, 8.0, 7.0])
    store.define(Field("FSPB", sentinel, "vec", "out", "newton"))
    forces.execute(vehicle, _ctx())
    np.testing.assert_array_equal(store.get("FSPB"), sentinel)
    assert "time" not in store.names()


def test_absent_farcs_fmrcs_treated_as_zero():
    vehicle, forces = _ready()
    store = vehicle.store
    assert "FARCS" not in store.names()
    assert "FMRCS" not in store.names()
    forces.execute(vehicle, _ctx())
    want_fapb = _cpp_fapb(
        pdynmc=PDYNMC, refa=REFA, ca=CA, cy=0.0, cn=0.0, thrust=THRUST
    )
    want_fmb = _cpp_fmb(
        pdynmc=PDYNMC, refa=REFA, refl=REFL, cll=0.0, clm=0.0, cln=0.0
    )
    np.testing.assert_allclose(store.get("FAPB"), want_fapb, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FMB"), want_fmb, rtol=RTOL, atol=ATOL)


def test_present_farcs_fmrcs_are_added():
    vehicle, forces = _ready()
    store = vehicle.store
    farcs = np.array([100.0, -20.0, 3.0])
    fmrcs = np.array([-4.0, 5.0, -6.0])
    store.define(Field("FARCS", farcs, "vec", "out", "rcs"))
    store.define(Field("FMRCS", fmrcs, "vec", "out", "rcs"))
    forces.execute(vehicle, _ctx())
    want_fapb = _cpp_fapb(
        pdynmc=PDYNMC,
        refa=REFA,
        ca=CA,
        cy=0.0,
        cn=0.0,
        thrust=THRUST,
        farcs=farcs,
    )
    want_fmb = _cpp_fmb(
        pdynmc=PDYNMC, refa=REFA, refl=REFL, cll=0.0, clm=0.0, cln=0.0, fmrcs=fmrcs
    )
    np.testing.assert_allclose(store.get("FAPB"), want_fapb, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FMB"), want_fmb, rtol=RTOL, atol=ATOL)


def test_mtvc_zero_does_not_add_fpb_fmpb():
    vehicle, forces = _ready(mtvc=0)
    store = vehicle.store
    store.define(Field("FPB", (50.0, 40.0, 30.0), "vec", "out", "tvc"))
    store.define(Field("FMPB", (7.0, 8.0, 9.0), "vec", "out", "tvc"))
    forces.execute(vehicle, _ctx())
    want_fapb = _cpp_fapb(
        pdynmc=PDYNMC, refa=REFA, ca=CA, cy=0.0, cn=0.0, thrust=THRUST
    )
    want_fmb = _cpp_fmb(
        pdynmc=PDYNMC, refa=REFA, refl=REFL, cll=0.0, clm=0.0, cln=0.0
    )
    np.testing.assert_allclose(store.get("FAPB"), want_fapb, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FMB"), want_fmb, rtol=RTOL, atol=ATOL)


def test_mtvc_absent_treated_as_zero():
    vehicle, forces = _ready(mtvc=None)
    assert "mtvc" not in vehicle.store.names()
    forces.execute(vehicle, _ctx())
    want = -PDYNMC * REFA * CA + THRUST
    np.testing.assert_allclose(
        vehicle.store.get("FAPB")[0], want, rtol=RTOL, atol=ATOL
    )


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.sam6.forces as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src
    from cadac.vehicles.plane6.forces import Plane6Forces

    assert not issubclass(Sam6Forces, Plane6Forces)


def test_terminate_exists_and_is_pass():
    vehicle, forces = _ready()
    assert forces.terminate(vehicle, _ctx()) is None
    np.testing.assert_array_equal(vehicle.store.get("FAPB"), np.zeros(3))
    np.testing.assert_array_equal(vehicle.store.get("FMB"), np.zeros(3))
    assert "FSPB" not in vehicle.store.names()

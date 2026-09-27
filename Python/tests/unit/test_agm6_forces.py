from types import SimpleNamespace

import numpy as np

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.agm6.forces import Agm6Forces

RTOL = 1e-12
ATOL = 1e-14

# Frozen aero/prop point from the AGM6 forces plan (Missile::forces)
PDYNMC = 12000.0
REFA = 0.196
REFL = 0.5
CA = 0.3
CY = 0.0
CN = 0.5
CLL = 0.0
CLM = -0.1
CLN = 0.0
THRUST = 10000.0

# C++ Missile::forces:
# FAPB[0]=-pdynmc*refa*ca+thrust
# FAPB[1]=pdynmc*refa*cy
# FAPB[2]=-pdynmc*refa*cn
# FMB=pdynmc*refa*refl*[cll,clm,cln]
# 12000*0.196 = 2352; 2352*0.5 = 1176
WANT_FAPB = np.array([9294.4, 0.0, -1176.0], dtype=float)
WANT_FMB = np.array([0.0, -117.6, 0.0], dtype=float)

DEFINED = ("FAPB", "FMB")
EXTERNALS = (
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
    "FSPB",
    "vmass",
    "time",
    "refb",
    "refc",
    "cx",
    "cz",
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


def _inputs(
    store,
    *,
    pdynmc=PDYNMC,
    refa=REFA,
    refl=REFL,
    ca=CA,
    cy=CY,
    cn=CN,
    cll=CLL,
    clm=CLM,
    cln=CLN,
    thrust=THRUST,
):
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("thrust", thrust, "real", "out", "propulsion"))
    store.define(Field("refa", refa, "real", "init", "aerodynamics"))
    store.define(Field("refl", refl, "real", "init", "aerodynamics"))
    store.define(Field("ca", ca, "real", "out", "aerodynamics"))
    store.define(Field("cy", cy, "real", "out", "aerodynamics"))
    store.define(Field("cn", cn, "real", "out", "aerodynamics"))
    store.define(Field("cll", cll, "real", "out", "aerodynamics"))
    store.define(Field("clm", clm, "real", "out", "aerodynamics"))
    store.define(Field("cln", cln, "real", "out", "aerodynamics"))


def _ready(**kwargs):
    vehicle = SimpleNamespace(store=StateStore())
    forces = Agm6Forces()
    forces.define(vehicle)
    _inputs(vehicle.store, **kwargs)
    forces.initialize(vehicle, _ctx())
    return vehicle, forces


def test_name_is_forces():
    assert Agm6Forces.name == "forces"
    assert Agm6Forces().name == "forces"


def test_define_registers_fapb_fmb_only():
    vehicle = SimpleNamespace(store=StateStore())
    Agm6Forces().define(vehicle)
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
    assert THRUST != 0.0
    assert WANT_FAPB[0] != THRUST
    assert WANT_FAPB[2] < 0.0
    assert WANT_FMB[1] < 0.0

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


def test_terminate_exists_and_is_pass():
    vehicle, forces = _ready()
    store = vehicle.store
    assert forces.terminate(vehicle, _ctx()) is None
    np.testing.assert_array_equal(store.get("FAPB"), np.zeros(3))
    np.testing.assert_array_equal(store.get("FMB"), np.zeros(3))
    assert "FSPB" not in store.names()

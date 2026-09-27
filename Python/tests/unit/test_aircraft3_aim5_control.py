import math

import numpy as np
import pytest

from cadac.constants import DEG, EPS, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat3.aim5.aircraft import Aim5AircraftControl, Aim5AircraftForces

RTOL = 1e-12
ATOL = 1e-14

GRAV = 9.8
INT_STEP = 0.002
PHILIMX = 60.0
ALPLIMX = 12.0
CLALPHA = 0.0523
WINGLOADING = 3247.0
PDYNMC = 20000.0
# 20000 * 0.0523 * 12 / 3247 = 12552 / 3247
ANLIMX = 12552.0 / 3247.0

CONTROL_FIELDS = {
    "phiav": ("real", "state", ()),
    "phiavd": ("real", "state", ()),
    "tphi": ("real", "data", ()),
    "philimx": ("real", "data", ()),
    "phiavx": ("real", "out", ()),
    "phiavcx": ("real", "diag", ()),
    "anx": ("real", "state", ()),
    "anxd": ("real", "state", ()),
    "tanx": ("real", "data", ()),
    "alplimx": ("real", "data", ()),
    "ancomx": ("real", "diag", ()),
    "clalpha": ("real", "data", ()),
    "wingloading": ("real", "data", ()),
    "phiavout": ("real", "out", ()),
}

EXTERNALS = ("grav", "pdynmc", "TVL", "acft_option", "ACOML")


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=INT_STEP,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _ready_control(
    *,
    acml=None,
    tvl=None,
    acft_option=0,
    tphi=0.0,
    tanx=0.0,
    philimx=PHILIMX,
    alplimx=ALPLIMX,
    clalpha=CLALPHA,
    wingloading=WINGLOADING,
    pdynmc=PDYNMC,
    grav=GRAV,
):
    vehicle = _Vehicle()
    control = Aim5AircraftControl()
    control.define(vehicle)
    store = vehicle.store
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("TVL", np.eye(3) if tvl is None else tvl, "mat", "out", "kinematics"))
    store.define(Field("acft_option", acft_option, "int", "data", "guidance"))
    if acml is None:
        acml = np.array([0.0, 0.0, -grav])
    store.define(Field("ACOML", acml, "vec", "out", "guidance"))
    store.set("tphi", tphi)
    store.set("tanx", tanx)
    store.set("philimx", philimx)
    store.set("alplimx", alplimx)
    store.set("clalpha", clalpha)
    store.set("wingloading", wingloading)
    control.initialize(vehicle, _ctx())
    return vehicle, control


def test_name_is_control():
    assert Aim5AircraftControl().name == "control"


def test_forces_class_still_present():
    assert Aim5AircraftForces().name == "forces"


def test_define_registers_control_fields():
    vehicle = _Vehicle()
    Aim5AircraftControl().define(vehicle)
    store = vehicle.store
    names = store.names()
    for name, (ftype, role, outputs) in CONTROL_FIELDS.items():
        assert name in names
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "control"
        assert field.outputs == outputs
        assert field.value == pytest.approx(0.0, abs=ATOL)
    for name in EXTERNALS:
        assert name not in names


def test_define_skips_existing_phiavout():
    vehicle = _Vehicle()
    store = vehicle.store
    store.define(Field("phiavout", 1.5, "real", "out", "pre", ("plot",)))
    Aim5AircraftControl().define(vehicle)
    field = store.field("phiavout")
    assert field.value == pytest.approx(1.5, abs=ATOL)
    assert field.module == "pre"
    assert field.role == "out"
    assert field.outputs == ("plot",)
    assert store.field("phiavx").module == "control"
    assert store.field("anx").module == "control"


def test_initialize_is_pass():
    vehicle, _control = _ready_control()
    store = vehicle.store
    assert store.get("phiav") == pytest.approx(0.0, abs=ATOL)
    assert store.get("phiavd") == pytest.approx(0.0, abs=ATOL)
    assert store.get("anx") == pytest.approx(0.0, abs=ATOL)
    assert store.get("anxd") == pytest.approx(0.0, abs=ATOL)
    assert store.get("phiavx") == pytest.approx(0.0, abs=ATOL)
    assert store.get("phiavout") == pytest.approx(0.0, abs=ATOL)


def test_terminate_is_pass():
    vehicle, control = _ready_control()
    control.terminate(vehicle, _ctx())
    store = vehicle.store
    assert store.get("phiav") == pytest.approx(0.0, abs=ATOL)
    assert store.get("anx") == pytest.approx(0.0, abs=ATOL)
    assert store.get("phiavx") == pytest.approx(0.0, abs=ATOL)


def test_hori_level_flight_unit_load_zero_bank():
    vehicle, control = _ready_control(
        acml=np.array([0.0, 0.0, -GRAV]),
        tvl=np.eye(3),
        acft_option=0,
        tphi=0.0,
        tanx=0.0,
        philimx=PHILIMX,
    )
    control.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("phiavx") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert store.get("phiavout") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert store.get("anx") == pytest.approx(1.0, rel=RTOL, abs=ATOL)
    assert store.get("phiavcx") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(1.0, rel=RTOL, abs=ATOL)


def test_lateral_acml_bank_under_philimx():
    vehicle, control = _ready_control(
        acml=np.array([0.0, GRAV, -GRAV]),
        tvl=np.eye(3),
        acft_option=0,
        tphi=0.0,
        tanx=0.0,
        philimx=PHILIMX,
    )
    control.execute(vehicle, _ctx())
    store = vehicle.store
    phiavc_deg = math.atan2(GRAV, GRAV) * DEG
    assert store.get("phiavcx") == pytest.approx(phiavc_deg, rel=RTOL, abs=ATOL)
    assert store.get("phiavcx") == pytest.approx(45.0, rel=RTOL, abs=ATOL)
    assert abs(store.get("phiavx")) <= PHILIMX
    assert store.get("phiavx") == pytest.approx(45.0, rel=RTOL, abs=ATOL)
    assert store.get("phiavout") == pytest.approx(45.0 * RAD, rel=RTOL, abs=ATOL)


def test_philimx_clips_large_bank():
    vehicle, control = _ready_control(
        acml=np.array([0.0, 10.0 * GRAV, -GRAV]),
        tvl=np.eye(3),
        acft_option=0,
        tphi=0.0,
        tanx=0.0,
        philimx=PHILIMX,
    )
    control.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("phiavx") == pytest.approx(PHILIMX, rel=RTOL, abs=ATOL)
    assert abs(store.get("phiavx")) <= PHILIMX
    assert store.get("phiavout") == pytest.approx(PHILIMX * RAD, rel=RTOL, abs=ATOL)


def test_eps_zero_guard_zeros_bank():
    vehicle, control = _ready_control(
        acml=np.array([GRAV, EPS / 2.0, EPS / 2.0]),
        tvl=np.eye(3),
        acft_option=0,
        tphi=0.0,
        tanx=0.0,
    )
    control.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("phiavx") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert store.get("phiavout") == pytest.approx(0.0, rel=RTOL, abs=ATOL)
    assert store.get("phiavcx") == pytest.approx(0.0, rel=RTOL, abs=ATOL)


def test_acft_option_1_clips_huge_anx():
    vehicle, control = _ready_control(
        acml=np.array([0.0, 0.0, -1000.0 * GRAV]),
        tvl=np.eye(3),
        acft_option=1,
        tphi=0.0,
        tanx=0.0,
        pdynmc=PDYNMC,
    )
    control.execute(vehicle, _ctx())
    store = vehicle.store
    assert abs(store.get("anx")) <= ANLIMX
    assert store.get("anx") == pytest.approx(ANLIMX, rel=RTOL, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(1000.0, rel=RTOL, abs=ATOL)


def test_acft_option_0_does_not_clip_huge_anx():
    vehicle, control = _ready_control(
        acml=np.array([0.0, 0.0, -1000.0 * GRAV]),
        tvl=np.eye(3),
        acft_option=0,
        tphi=0.0,
        tanx=0.0,
        pdynmc=PDYNMC,
    )
    control.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("anx") == pytest.approx(1000.0, rel=RTOL, abs=ATOL)
    assert store.get("anx") > ANLIMX
    assert store.get("ancomx") == pytest.approx(1000.0, rel=RTOL, abs=ATOL)


def test_tphi_lags_bank_one_step():
    tphi = 0.1
    vehicle, control = _ready_control(
        acml=np.array([0.0, GRAV, -GRAV]),
        tvl=np.eye(3),
        acft_option=0,
        tphi=tphi,
        tanx=0.0,
    )
    control.execute(vehicle, _ctx())
    phiavc = math.atan2(GRAV, GRAV)
    phiavd_new = (phiavc - 0.0) / tphi
    phiav = integrate(phiavd_new, 0.0, 0.0, INT_STEP)
    store = vehicle.store
    assert store.get("phiav") == pytest.approx(phiav, rel=RTOL, abs=ATOL)
    assert store.get("phiavd") == pytest.approx(phiavd_new, rel=RTOL, abs=ATOL)
    assert store.get("phiavx") == pytest.approx(phiav * DEG, rel=RTOL, abs=ATOL)


def test_tanx_lags_load_one_step():
    tanx = 0.1
    vehicle, control = _ready_control(
        acml=np.array([0.0, GRAV, -GRAV]),
        tvl=np.eye(3),
        acft_option=0,
        tphi=0.0,
        tanx=tanx,
    )
    control.execute(vehicle, _ctx())
    ancomx = math.sqrt(GRAV * GRAV + GRAV * GRAV) / GRAV
    anxd_new = (ancomx - 0.0) / tanx
    anx = integrate(anxd_new, 0.0, 0.0, INT_STEP)
    store = vehicle.store
    assert store.get("ancomx") == pytest.approx(ancomx, rel=RTOL, abs=ATOL)
    assert store.get("anx") == pytest.approx(anx, rel=RTOL, abs=ATOL)
    assert store.get("anxd") == pytest.approx(anxd_new, rel=RTOL, abs=ATOL)

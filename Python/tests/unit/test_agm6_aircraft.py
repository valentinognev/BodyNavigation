import inspect
from math import atan2, fabs, sqrt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.cli import _VEHICLE_FAMILIES
from cadac.constants import DEG, EPS, RAD
from cadac.eom.flat3 import Flat3Kinematics
from cadac.env.gravity import gravity
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.agm6.aircraft import (
    Agm6Aircraft,
    Agm6AircraftControl,
    Agm6AircraftForces,
    Agm6AircraftGuidance,
    Agm6AircraftSensor,
)
from cadac.vehicles.flat6.agm6.flat3io import Agm6Flat3Environment, Agm6Flat3Newton

RTOL = 1e-12
ATOL = 1e-14
ZEROS3 = (0.0, 0.0, 0.0)
ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
GRAV = 9.81
GTURN = 1.5
TPHI = 0.5
TANX = 0.5
PHILIMX = 60.0
INT_STEP = 0.001
SAEL3 = -7000.0
DVAE = 293.0
ALT = -SAEL3
VEHICLE_GRAV = gravity(ALT)

GUIDANCE_FIELDS = {
    "acft_option": ("int", "data", 0, ()),
    "guid_gain": ("real", "data", 0.0, ()),
    "ACOML": ("vec", "out", ZEROS3, ()),
    "gturn": ("real", "data", 0.0, ()),
}

CONTROL_FIELDS = {
    "phiav": ("real", "state", 0.0, ()),
    "phiavd": ("real", "state", 0.0, ()),
    "tphi": ("real", "data", 0.0, ()),
    "philimx": ("real", "data", 0.0, ()),
    "phiavx": ("real", "out", 0.0, ()),
    "phiavcx": ("real", "diag", 0.0, ()),
    "anx": ("real", "state", 0.0, ()),
    "anxd": ("real", "state", 0.0, ()),
    "tanx": ("real", "data", 0.0, ()),
    "alplimx": ("real", "data", 0.0, ()),
    "ancomx": ("real", "diag", 0.0, ()),
    "clalpha": ("real", "data", 0.0, ()),
    "wingloading": ("real", "data", 0.0, ()),
    "phiavout": ("real", "out", 0.0, ()),
}

FORCES_FIELDS = {
    "FSPA": ("vec", "out", ZEROS3, ()),
    "acc_longx": ("real", "data", 0.0, ()),
}

SENSOR_FIELDS = {
    "init_flag": ("int", "data", 1, ()),
    "track_epoch": ("real", "save", 0.0, ()),
    "track_step": ("real", "data", 0.0, ()),
    "target_num": ("int", "save", 0, ()),
    "STCEL1": ("vec", "out", ZEROS3, ("com",)),
    "VTCEL1": ("vec", "out", ZEROS3, ("com",)),
    "STCEL2": ("vec", "out", ZEROS3, ("com",)),
    "VTCEL2": ("vec", "out", ZEROS3, ("com",)),
    "STCEL3": ("vec", "out", ZEROS3, ("com",)),
    "VTCEL3": ("vec", "out", ZEROS3, ("com",)),
    "STCEL4": ("vec", "out", ZEROS3, ()),
    "VTCEL4": ("vec", "out", ZEROS3, ()),
    "STCEL5": ("vec", "out", ZEROS3, ()),
    "VTCEL5": ("vec", "out", ZEROS3, ()),
    "dat_sigma": ("real", "data", 0.0, ()),
    "azat_sigma": ("real", "data", 0.0, ()),
    "elat_sigma": ("real", "data", 0.0, ()),
    "vel_sigma": ("real", "data", 0.0, ()),
}

GUIDANCE_EXTERNALS = ("grav", "TVL", "SAEL", "VAEL")
CONTROL_EXTERNALS = ("grav", "pdynmc", "TVL", "acft_option", "ACOML")
FORCES_EXTERNALS = ("grav", "anx")


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def _skew(vec):
    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def _ctx(int_step=INT_STEP, combus=None):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=0,
    )


def _define_externals(store, names):
    for name in names:
        if name in store.names():
            continue
        if name == "TVL":
            store.define(Field(name, ZEROS33, "mat", "diag", "newton"))
        elif name in ("SAEL", "VAEL", "ACOML"):
            store.define(Field(name, ZEROS3, "vec", "out", "newton"))
        elif name == "acft_option":
            store.define(Field(name, 0, "int", "data", "guidance"))
        else:
            store.define(Field(name, 0.0, "real", "out", "environment"))


def _ready_guidance(*, acft_option=0, gturn=0.0, guid_gain=0.0, combus=None):
    vehicle = SimpleNamespace(store=StateStore())
    guidance = Agm6AircraftGuidance()
    guidance.define(vehicle)
    _define_externals(vehicle.store, GUIDANCE_EXTERNALS)
    vehicle.store.set("acft_option", acft_option)
    vehicle.store.set("gturn", gturn)
    vehicle.store.set("guid_gain", guid_gain)
    vehicle.store.set("grav", GRAV)
    vehicle.store.set("TVL", np.eye(3))
    vehicle.store.set("SAEL", np.array([0.0, 0.0, SAEL3]))
    vehicle.store.set("VAEL", np.array([DVAE, 0.0, 0.0]))
    return vehicle, guidance, _ctx(combus=combus)


def _control_cpp(store, int_step):
    tphi = store.get("tphi")
    philimx = store.get("philimx")
    tanx = store.get("tanx")
    alplimx = store.get("alplimx")
    clalpha = store.get("clalpha")
    wingloading = store.get("wingloading")
    grav = store.get("grav")
    pdynmc = store.get("pdynmc")
    tvl = np.asarray(store.get("TVL"), dtype=float)
    acft_option = store.get("acft_option")
    acoml = np.asarray(store.get("ACOML"), dtype=float)
    phiav = store.get("phiav")
    phiavd = store.get("phiavd")
    anx = store.get("anx")
    anxd = store.get("anxd")
    acomv = tvl @ acoml
    acoma2 = acomv[1]
    acoma3 = acomv[2]
    if fabs(acoma2) < EPS and fabs(acoma3) < EPS:
        phiavc = 0.0
    else:
        phiavc = atan2(acoma2, -acoma3)
    phiavcx = phiavc * DEG
    if tphi:
        phiavd_new = (phiavc - phiav) / tphi
        phiav = integrate(phiavd_new, phiavd, phiav, int_step)
        phiavd = phiavd_new
    else:
        phiav = phiavc
    phiavx = phiav * DEG
    if fabs(phiavx) >= philimx:
        phiavx = philimx * _sign(phiavx)
    phiavout = phiavx * RAD
    ancomx = sqrt(acoma2 * acoma2 + acoma3 * acoma3) / grav
    if tanx:
        anxd_new = (ancomx - anx) / tanx
        anx = integrate(anxd_new, anxd, anx, int_step)
        anxd = anxd_new
    else:
        anx = ancomx
    if acft_option > 0:
        anlimx = pdynmc * clalpha * alplimx / wingloading
        if fabs(anx) >= anlimx:
            anx = anlimx * _sign(anx)
    return phiav, phiavd, phiavx, phiavcx, phiavout, ancomx, anx, anxd


def _ready_control(*, acft_option=0, acoml=None, tphi=TPHI, tanx=TANX, philimx=PHILIMX):
    vehicle = SimpleNamespace(store=StateStore())
    control = Agm6AircraftControl()
    control.define(vehicle)
    _define_externals(vehicle.store, CONTROL_EXTERNALS)
    if acoml is None:
        acoml = np.array([0.0, 0.0, -GRAV])
    vehicle.store.set("acft_option", acft_option)
    vehicle.store.set("tphi", tphi)
    vehicle.store.set("tanx", tanx)
    vehicle.store.set("philimx", philimx)
    vehicle.store.set("alplimx", 12.0)
    vehicle.store.set("clalpha", 0.0523)
    vehicle.store.set("wingloading", 3247.0)
    vehicle.store.set("grav", GRAV)
    vehicle.store.set("pdynmc", 50000.0)
    vehicle.store.set("TVL", np.eye(3))
    vehicle.store.set("ACOML", acoml)
    return vehicle, control, _ctx()


def _ready_forces(*, anx=0.0, acc_longx=0.0):
    vehicle = SimpleNamespace(store=StateStore())
    forces = Agm6AircraftForces()
    forces.define(vehicle)
    _define_externals(vehicle.store, FORCES_EXTERNALS)
    vehicle.store.set("grav", GRAV)
    vehicle.store.set("anx", anx)
    vehicle.store.set("acc_longx", acc_longx)
    return vehicle, forces, _ctx()


def _plant_ics(store):
    store.set("sael1", 0.0)
    store.set("sael2", 0.0)
    store.set("sael3", SAEL3)
    store.set("dvae", DVAE)
    store.set("psivlx", 0.0)
    store.set("thtvlx", 0.0)
    store.set("acft_option", 0)
    store.set("gturn", GTURN)
    store.set("tphi", TPHI)
    store.set("tanx", TANX)
    store.set("philimx", PHILIMX)
    store.set("alplimx", 12.0)
    store.set("clalpha", 0.0523)
    store.set("wingloading", 3247.0)
    store.set("acc_longx", 0.0)


def test_guidance_name_is_guidance():
    assert Agm6AircraftGuidance.name == "guidance"
    assert Agm6AircraftGuidance().name == "guidance"


def test_control_name_is_control():
    assert Agm6AircraftControl.name == "control"
    assert Agm6AircraftControl().name == "control"


def test_forces_name_is_forces():
    assert Agm6AircraftForces.name == "forces"
    assert Agm6AircraftForces().name == "forces"


def test_sensor_name_is_sensor():
    assert Agm6AircraftSensor.name == "sensor"
    assert Agm6AircraftSensor().name == "sensor"


def test_guidance_define_registers_cpp_fields():
    vehicle = SimpleNamespace(store=StateStore())
    Agm6AircraftGuidance().define(vehicle)
    for name, (ftype, role, default, outputs) in GUIDANCE_FIELDS.items():
        field = vehicle.store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "guidance"
        assert field.outputs == outputs
        if ftype == "int":
            assert vehicle.store.get(name) == default
        elif ftype == "real":
            assert vehicle.store.get(name) == default
        else:
            np.testing.assert_array_equal(vehicle.store.get(name), np.zeros(3))


def test_control_define_registers_cpp_fields():
    vehicle = SimpleNamespace(store=StateStore())
    Agm6AircraftControl().define(vehicle)
    for name, (ftype, role, default, outputs) in CONTROL_FIELDS.items():
        field = vehicle.store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "control"
        assert field.outputs == outputs
        assert vehicle.store.get(name) == default


def test_forces_define_registers_cpp_fields_and_fspv():
    vehicle = SimpleNamespace(store=StateStore())
    Agm6AircraftForces().define(vehicle)
    store = vehicle.store
    for name, (ftype, role, default, outputs) in FORCES_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "forces"
        assert field.outputs == outputs
        if ftype == "real":
            assert store.get(name) == default
        else:
            np.testing.assert_array_equal(store.get(name), np.zeros(3))
    assert store.field("FSPV").type == "vec"
    assert store.field("FSPV").module == "forces"


def test_sensor_define_registers_cpp_def_sensor_fields():
    vehicle = SimpleNamespace(store=StateStore())
    Agm6AircraftSensor().define(vehicle)
    store = vehicle.store
    assert "track_on" not in store.names()
    for name, (ftype, role, default, outputs) in SENSOR_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "sensor"
        assert field.outputs == outputs
        if ftype == "vec":
            np.testing.assert_array_equal(store.get(name), np.zeros(3))
        else:
            assert store.get(name) == default


def test_acft_option_0_acoml_gravity_bias():
    vehicle, guidance, ctx = _ready_guidance(acft_option=0)
    guidance.execute(vehicle, ctx)
    want = np.array([0.0, 0.0, -GRAV])
    np.testing.assert_allclose(vehicle.store.get("ACOML"), want, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("ACOML")[2], -GRAV, rtol=RTOL, atol=ATOL)


def test_acft_option_1_horizontal_g_turn():
    vehicle, guidance, ctx = _ready_guidance(acft_option=1, gturn=GTURN)
    tvl = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    vehicle.store.set("TVL", tvl)
    guidance.execute(vehicle, ctx)
    acomv = np.array([0.0, GTURN * GRAV, -GRAV])
    want = tvl.T @ acomv
    np.testing.assert_allclose(vehicle.store.get("ACOML"), want, rtol=RTOL, atol=ATOL)


def test_acft_option_2_escape_uses_first_target3_packet():
    sael_tgt = np.array([33000.0, 10000.0, -100.0])
    vael_tgt = np.array([0.0, -5.0, 0.0])
    decoy = np.array([1.0, 2.0, 3.0])
    combus = [
        Packet(name="m1", type="MISSILE6", status=1, vars={"SAEL": decoy, "VAEL": decoy}),
        Packet(
            name="t2",
            type="TARGET3",
            status=1,
            vars={"SAEL": sael_tgt, "VAEL": vael_tgt},
        ),
        Packet(
            name="t1",
            type="TARGET3",
            status=1,
            vars={"SAEL": decoy, "VAEL": decoy},
        ),
    ]
    guid_gain = 1.0
    vehicle, guidance, ctx = _ready_guidance(
        acft_option=2, guid_gain=guid_gain, combus=combus
    )
    sael = np.array([0.0, 0.0, SAEL3])
    vael = np.array([DVAE, 0.0, 0.0])
    vehicle.store.set("SAEL", sael)
    vehicle.store.set("VAEL", vael)
    guidance.execute(vehicle, ctx)
    satl = sael - sael_tgt
    dab = float(np.linalg.norm(satl))
    dum = float(np.linalg.norm(_skew(vael) @ vael_tgt))
    gain = guid_gain * dum / dab
    uvtel = vael_tgt / np.linalg.norm(vael_tgt)
    uvael = vael / np.linalg.norm(vael)
    epsl = _skew(uvael) @ uvtel
    want = _skew(epsl) @ uvael * gain + np.array([0.0, 0.0, -GRAV])
    np.testing.assert_allclose(vehicle.store.get("ACOML"), want, rtol=RTOL, atol=ATOL)


def test_acft_option_3_raises():
    vehicle, guidance, ctx = _ready_guidance(acft_option=3)
    with pytest.raises(ValueError):
        guidance.execute(vehicle, ctx)


def test_control_one_step_tphi_lag_finite_phiavout():
    acoml = np.array([0.0, GTURN * GRAV, -GRAV])
    vehicle, control, ctx = _ready_control(acoml=acoml)
    want = _control_cpp(vehicle.store, ctx.int_step)
    control.execute(vehicle, ctx)
    store = vehicle.store
    assert np.isfinite(store.get("phiavout"))
    np.testing.assert_allclose(store.get("phiav"), want[0], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("phiavd"), want[1], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("phiavx"), want[2], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("phiavcx"), want[3], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("phiavout"), want[4], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ancomx"), want[5], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("anx"), want[6], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("anxd"), want[7], rtol=RTOL, atol=ATOL)


def test_control_bank_limit_and_zero_lags():
    acoml = np.array([0.0, 10.0 * GRAV, -GRAV])
    vehicle, control, ctx = _ready_control(acoml=acoml, tphi=0.0, tanx=0.0, philimx=60.0)
    want = _control_cpp(vehicle.store, ctx.int_step)
    control.execute(vehicle, ctx)
    np.testing.assert_allclose(vehicle.store.get("phiavx"), 60.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        vehicle.store.get("phiavout"), want[4], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(vehicle.store.get("anx"), want[6], rtol=RTOL, atol=ATOL)


def test_forces_fspa_and_fspv():
    anx = 0.001
    acc_longx = 0.2
    vehicle, forces, ctx = _ready_forces(anx=anx, acc_longx=acc_longx)
    forces.execute(vehicle, ctx)
    want = np.array([acc_longx * GRAV, 0.0, -anx * GRAV])
    np.testing.assert_allclose(vehicle.store.get("FSPA"), want, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("FSPV"), want, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        vehicle.store.get("FSPA")[2], -anx * GRAV, rtol=RTOL, atol=ATOL
    )


def test_sensor_empty_combus_leaves_stcel1_zeros():
    vehicle = SimpleNamespace(store=StateStore())
    sensor = Agm6AircraftSensor()
    sensor.define(vehicle)
    vehicle.store.define(Field("SAEL", ZEROS3, "vec", "state", "newton", ("com",)))
    before = np.array(vehicle.store.get("STCEL1"), copy=True)
    sensor.initialize(vehicle, _ctx())
    sensor.execute(vehicle, _ctx())
    sensor.terminate(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("STCEL1"), before)


def test_aircraft_type_constructor_modules_and_com_names():
    sig = inspect.signature(Agm6Aircraft.__init__)
    assert list(sig.parameters) == ["self", "name", "events"]
    assert sig.parameters["events"].default is None
    vehicle = Agm6Aircraft("Blue")
    assert vehicle.type == "AIRCRAFT3"
    assert vehicle.name == "Blue"
    assert vehicle.health == 1
    assert [type(m) for m in vehicle.modules] == [
        Agm6Flat3Environment,
        Flat3Kinematics,
        Agm6AircraftGuidance,
        Agm6AircraftControl,
        Agm6AircraftForces,
        Agm6Flat3Newton,
        Agm6AircraftSensor,
    ]
    vehicle.define()
    for name in ("SAEL", "VAEL", "STCEL1", "VTCEL1"):
        assert name in vehicle.com_names
        assert "com" in vehicle.store.field(name).outputs


def test_define_param_set_initialize_and_one_execute_option_0():
    vehicle = Agm6Aircraft("Blue")
    vehicle.define()
    _plant_ics(vehicle.store)
    ctx = _ctx()
    for module in vehicle.modules:
        module.initialize(vehicle, ctx)
    np.testing.assert_allclose(
        vehicle.store.get("SAEL"),
        np.array([0.0, 0.0, SAEL3]),
        rtol=RTOL,
        atol=ATOL,
    )
    for module in vehicle.modules:
        module.execute(vehicle, ctx)
    grav = vehicle.store.get("grav")
    np.testing.assert_allclose(grav, VEHICLE_GRAV, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        vehicle.store.get("ACOML")[2], -grav, rtol=RTOL, atol=ATOL
    )
    assert np.isfinite(vehicle.store.get("phiavout"))
    anx = vehicle.store.get("anx")
    np.testing.assert_allclose(
        vehicle.store.get("FSPA")[2], -anx * grav, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("FSPV"), vehicle.store.get("FSPA"), rtol=RTOL, atol=ATOL
    )


def test_hyper5_plane6_sam6_not_imported():
    import cadac.vehicles.flat6.agm6.aircraft as aircraft_mod

    src = Path(aircraft_mod.__file__).read_text(encoding="utf-8")
    lower = src.lower()
    assert "hyper5" not in lower
    assert "plane6" not in lower
    assert "sam6" not in lower
    assert "from cadac.vehicles.round3.hyper5" not in src
    assert "from cadac.vehicles.flat6.falcon6" not in src
    assert "from cadac.vehicles.flat6.sam6" not in src


def test_aircraft3_registered_in_families():
    from cadac.vehicles.flat6.agm6.aircraft import Agm6Aircraft

    assert _VEHICLE_FAMILIES[("agm6", "AIRCRAFT3")] is Agm6Aircraft

import math
from pathlib import Path

import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.eom.flat6 import Flat6Kinematics
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat3tr
from cadac.vehicles.flat6.sam6.kinematics import Sam6Kinematics

RTOL = 1e-12
ATOL = 1e-14
SMALL = 1e-7

DEFINED = (
    "time",
    "event_time",
    "int_step_new",
    "out_step_fact",
    "stop",
    "lconv",
    "launch_delay",
    "launch_epoch",
    "launch_time",
    "msl_time",
    "ck",
    "q0d",
    "q0",
    "q1d",
    "q1",
    "q2d",
    "q2",
    "q3d",
    "q3",
    "TBL",
    "psibl",
    "thtbl",
    "phibl",
    "psiblx",
    "thtblx",
    "phiblx",
    "alppx",
    "phipx",
    "alpp",
    "phip",
    "alphax",
    "betax",
    "ortho_error",
    "etbl",
    "TLB",
)
ROLES = {
    "time": "exec",
    "event_time": "exec",
    "int_step_new": "exec",
    "out_step_fact": "exec",
    "stop": "exec",
    "lconv": "exec",
    "launch_delay": "exec",
    "launch_epoch": "exec",
    "launch_time": "exec",
    "msl_time": "exec",
    "ck": "data",
    "q0d": "state",
    "q0": "state",
    "q1d": "state",
    "q1": "state",
    "q2d": "state",
    "q2": "state",
    "q3d": "state",
    "q3": "state",
    "TBL": "out",
    "psibl": "diag",
    "thtbl": "diag",
    "phibl": "diag",
    "psiblx": "in/diag",
    "thtblx": "in/diag",
    "phiblx": "in/diag",
    "alppx": "out",
    "phipx": "out",
    "alpp": "out",
    "phip": "out",
    "alphax": "diag",
    "betax": "diag",
    "ortho_error": "diag",
    "etbl": "diag",
    "TLB": "diag",
}
OUTPUTS = {
    "time": ("scrn", "plot", "com"),
    "event_time": (),
    "int_step_new": (),
    "out_step_fact": (),
    "stop": (),
    "lconv": ("plot",),
    "launch_delay": (),
    "launch_epoch": (),
    "launch_time": (),
    "msl_time": ("plot",),
    "ck": (),
    "q0d": (),
    "q0": (),
    "q1d": (),
    "q1": (),
    "q2d": (),
    "q2": (),
    "q3d": (),
    "q3": (),
    "TBL": (),
    "psibl": (),
    "thtbl": (),
    "phibl": (),
    "psiblx": ("scrn", "plot"),
    "thtblx": ("scrn", "plot"),
    "phiblx": ("scrn", "plot"),
    "alppx": ("plot",),
    "phipx": ("plot",),
    "alpp": (),
    "phip": (),
    "alphax": ("scrn", "plot"),
    "betax": ("scrn", "plot"),
    "ortho_error": ("scrn",),
    "etbl": (),
    "TLB": (),
}
INT_FIELDS = ("stop", "lconv")
MAT_FIELDS = ("TBL", "TLB")
NOT_DEFINED = (
    "VBEB",
    "VBAL",
    "dvba",
    "dvbe",
    "WBEB",
    "pp",
    "qq",
    "rr",
    "trcond",
    "trortho",
    "tralp",
    "hbe",
    "VBEL",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(
    sim_time=0.0,
    int_step=0.001,
    event_time=0.0,
    out_fact=0.0,
    combus=None,
    vehicle_slot=0,
):
    return SimContext(
        sim_time=sim_time,
        int_step=int_step,
        event_time=event_time,
        out_fact=out_fact,
        combus=combus,
        vehicle_slot=vehicle_slot,
    )


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _cpp_quat(psiblx, thtblx, phiblx):
    spsi = math.sin(psiblx / (2.0 * DEG))
    cpsi = math.cos(psiblx / (2.0 * DEG))
    stht = math.sin(thtblx / (2.0 * DEG))
    ctht = math.cos(thtblx / (2.0 * DEG))
    sphi = math.sin(phiblx / (2.0 * DEG))
    cphi = math.cos(phiblx / (2.0 * DEG))
    q0 = cpsi * ctht * cphi + spsi * stht * sphi
    q1 = cpsi * ctht * sphi - spsi * stht * cphi
    q2 = cpsi * stht * cphi + spsi * ctht * sphi
    q3 = -cpsi * stht * sphi + spsi * ctht * cphi
    return q0, q1, q2, q3


def _plant(store, vbeb=(16.0, 0.0, 0.0), pp=0.0, qq=0.0, rr=0.0):
    if "VBEB" not in store.names():
        store.define(Field("VBEB", (0.0, 0.0, 0.0), "vec", "state", "newton"))
    if "pp" not in store.names():
        store.define(Field("pp", 0.0, "real", "state", "euler"))
        store.define(Field("qq", 0.0, "real", "state", "euler"))
        store.define(Field("rr", 0.0, "real", "state", "euler"))
    store.set("VBEB", np.asarray(vbeb, dtype=float))
    store.set("pp", float(pp))
    store.set("qq", float(qq))
    store.set("rr", float(rr))


def _ready(launch_delay=0.0, vbeb=(16.0, 0.0, 0.0), **ctx_kw):
    vehicle = _Vehicle()
    kin = Sam6Kinematics()
    kin.define(vehicle)
    vehicle.store.set("launch_delay", launch_delay)
    init_ctx = _ctx(sim_time=0.0, int_step=ctx_kw.get("int_step", 0.001))
    kin.initialize(vehicle, init_ctx)
    _plant(vehicle.store, vbeb=vbeb)
    return vehicle, kin, _ctx(**ctx_kw)


def test_name_is_kinematics():
    assert Sam6Kinematics().name == "kinematics"


def test_define_cpp_fields():
    vehicle = _Vehicle()
    Sam6Kinematics().define(vehicle)
    store = vehicle.store
    assert tuple(store.names()) == DEFINED
    for name in DEFINED:
        field = store.field(name)
        assert field.module == "kinematics"
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        if name in INT_FIELDS:
            assert field.type == "int"
            assert store.get(name) == 0
        elif name in MAT_FIELDS:
            assert field.type == "mat"
            np.testing.assert_allclose(store.get(name), np.zeros((3, 3)), atol=ATOL)
        elif name == "launch_delay":
            assert field.type == "real"
            assert _approx(store.get(name), 99999.0)
        elif name == "ck":
            assert field.type == "real"
            assert _approx(store.get(name), 50.0)
        else:
            assert field.type == "real"
            assert _approx(store.get(name), 0.0), name


def test_define_does_not_register_newton_euler_or_aero_fields():
    vehicle = _Vehicle()
    Sam6Kinematics().define(vehicle)
    for name in NOT_DEFINED:
        assert name not in vehicle.store.names()


def test_initialize_launch_epoch_quats_and_tbl():
    vehicle = _Vehicle()
    kin = Sam6Kinematics()
    kin.define(vehicle)
    store = vehicle.store
    store.set("launch_delay", 12.0)
    store.set("psiblx", 10.0)
    store.set("thtblx", 80.0)
    store.set("phiblx", -5.0)
    kin.initialize(vehicle, _ctx(sim_time=0.0, int_step=0.001))
    assert _approx(store.get("launch_epoch"), 12.0)
    assert _approx(store.get("time"), 0.0)
    assert _approx(store.get("int_step_new"), 0.001)
    q0, q1, q2, q3 = _cpp_quat(10.0, 80.0, -5.0)
    assert store.get("q0") == q0
    assert store.get("q1") == q1
    assert store.get("q2") == q2
    assert store.get("q3") == q3
    np.testing.assert_allclose(
        store.get("TBL"),
        mat3tr(10.0 / DEG, 80.0 / DEG, -5.0 / DEG),
        rtol=RTOL,
        atol=ATOL,
    )
    assert store.get("alphax") == 0.0
    assert store.get("msl_time") == 0.0


def test_no_radar_msl_time_is_sim_time():
    vehicle, kin, ctx = _ready(launch_delay=0.0, sim_time=5.0, combus=None)
    kin.execute(vehicle, ctx)
    assert _approx(vehicle.store.get("msl_time"), 5.0)
    assert _approx(vehicle.store.get("time"), 5.0)


def test_empty_combus_msl_time_is_sim_time():
    vehicle, kin, ctx = _ready(launch_delay=0.0, sim_time=5.0, combus=[])
    kin.execute(vehicle, ctx)
    assert _approx(vehicle.store.get("msl_time"), 5.0)


def test_no_radar_does_not_use_kinematics_launch_delay():
    vehicle, kin, ctx = _ready(launch_delay=99999.0, sim_time=5.0, combus=None)
    kin.execute(vehicle, ctx)
    assert _approx(vehicle.store.get("msl_time"), 5.0)


def test_radar_first_missile_uses_lnch_delay_m1():
    combus = [
        Packet(name="SAM", type="MISSILE6", status=1, vars={}),
        Packet(
            name="R",
            type="RADAR0",
            status=1,
            vars={"lnch_delay_m1": 2.0, "lnch_delay_m2": 9.0, "lnch_delay_m3": 9.0},
        ),
    ]
    vehicle, kin, ctx = _ready(sim_time=5.0, combus=combus, vehicle_slot=0)
    kin.execute(vehicle, ctx)
    assert _approx(vehicle.store.get("msl_time"), 3.0)


def test_msl_time_indexes_among_missile6_not_combus_slot():
    combus = [
        Packet(
            name="f1",
            type="RADAR0",
            status=1,
            vars={"lnch_delay_m1": 2.0, "lnch_delay_m2": 4.0},
        ),
        Packet(name="M1", type="MISSILE6", status=1, vars={}),
        Packet(name="M2", type="MISSILE6", status=1, vars={}),
    ]
    vehicle, kin, ctx = _ready(sim_time=5.0, combus=combus, vehicle_slot=1)
    kin.execute(vehicle, ctx)
    assert _approx(vehicle.store.get("msl_time"), 3.0)
    vehicle2, kin2, ctx2 = _ready(sim_time=5.0, combus=combus, vehicle_slot=2)
    kin2.execute(vehicle2, ctx2)
    assert _approx(vehicle2.store.get("msl_time"), 1.0)


def test_identifies_radar_by_type_not_cpp_id():
    combus = [
        Packet(name="f1", type="AIRCRAFT3", status=1, vars={"lnch_delay_m1": 100.0}),
        Packet(name="other", type="RADAR0", status=1, vars={"lnch_delay_m1": 2.0}),
        Packet(name="m1", type="MISSILE6", status=1, vars={}),
    ]
    vehicle, kin, ctx = _ready(sim_time=5.0, combus=combus, vehicle_slot=2)
    kin.execute(vehicle, ctx)
    assert _approx(vehicle.store.get("msl_time"), 3.0)


def test_msl_time_clamped_at_zero_before_launch():
    combus = [
        Packet(name="SAM", type="MISSILE6", status=1, vars={}),
        Packet(name="R", type="RADAR0", status=1, vars={"lnch_delay_m1": 8.0}),
    ]
    vehicle, kin, ctx = _ready(sim_time=5.0, combus=combus, vehicle_slot=0)
    kin.execute(vehicle, ctx)
    assert _approx(vehicle.store.get("msl_time"), 0.0)


def test_identity_vbeb_zero_incidence():
    vehicle, kin, ctx = _ready(vbeb=(16.0, 0.0, 0.0), sim_time=0.0)
    kin.execute(vehicle, ctx)
    assert _approx(vehicle.store.get("alphax"), 0.0)
    assert _approx(vehicle.store.get("betax"), 0.0)


def test_vbeb_pitch_gives_alphax_10():
    vbeb = (16.0, 0.0, 16.0 * math.tan(10.0 * RAD))
    vehicle, kin, ctx = _ready(vbeb=vbeb, sim_time=0.0)
    kin.execute(vehicle, ctx)
    assert vehicle.store.get("alphax") == pytest.approx(10.0, rel=RTOL, abs=ATOL)
    assert _approx(vehicle.store.get("betax"), 0.0)


def test_incidence_reads_vbeb_not_vbal():
    vbeb = (16.0, 0.0, 16.0 * math.tan(10.0 * RAD))
    vehicle, kin, ctx = _ready(vbeb=vbeb, sim_time=0.0)
    store = vehicle.store
    store.define(Field("VBAL", (16.0, 0.0, 0.0), "vec", "out", "environment"))
    store.define(Field("dvba", 16.0, "real", "out", "environment"))
    kin.execute(vehicle, ctx)
    assert vehicle.store.get("alphax") == pytest.approx(10.0, rel=RTOL, abs=ATOL)


def test_trortho_tiny_and_large_ortho_error_sets_trcond_1():
    vehicle, kin, ctx = _ready(vbeb=(16.0, 0.0, 0.0), sim_time=0.0)
    store = vehicle.store
    store.define(Field("trcond", 0, "int", "diag", "aerodynamics"))
    store.define(Field("trortho", 1e-20, "real", "data", "aerodynamics"))
    store.define(Field("tralp", 1.047, "real", "data", "aerodynamics"))
    store.set("q0", 0.0)
    store.set("q1", 0.0)
    store.set("q2", 0.0)
    store.set("q3", 0.0)
    kin.execute(vehicle, ctx)
    ortho = 1.0 - (
        store.get("q0") ** 2
        + store.get("q1") ** 2
        + store.get("q2") ** 2
        + store.get("q3") ** 2
    )
    assert abs(store.get("ortho_error")) > store.get("trortho")
    assert abs(ortho) > store.get("trortho")
    assert store.get("trcond") == 1


def test_alpp_above_tralp_sets_trcond_2():
    vehicle, kin, ctx = _ready(vbeb=(0.0, 0.0, 16.0), sim_time=0.0)
    store = vehicle.store
    store.define(Field("trcond", 0, "int", "diag", "aerodynamics"))
    store.define(Field("trortho", 1.0, "real", "data", "aerodynamics"))
    store.define(Field("tralp", 1.047, "real", "data", "aerodynamics"))
    kin.execute(vehicle, ctx)
    assert store.get("alpp") > store.get("tralp")
    assert store.get("trcond") == 2


def test_skips_trcond_when_absent():
    vehicle, kin, ctx = _ready(sim_time=0.0)
    kin.execute(vehicle, ctx)
    assert "trcond" not in vehicle.store.names()
    assert np.isfinite(vehicle.store.get("alphax"))


def test_not_a_subclass_of_flat6_kinematics():
    assert not issubclass(Sam6Kinematics, Flat6Kinematics)
    import cadac.vehicles.flat6.sam6.kinematics as kinmod

    assert "Flat6Kinematics" not in dir(kinmod)


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.flat6.sam6.kinematics as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6Kinematics" not in src
    assert "from cadac.eom.flat6 import" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src


def test_int_step_new_updates_ctx_and_quaternion_integrate():
    vehicle, kin, ctx = _ready(sim_time=0.0, int_step=0.001)
    store = vehicle.store
    store.set("int_step_new", 0.01)
    store.set("pp", 1.0)
    ctx = _ctx(sim_time=0.0, int_step=0.001)
    kin.execute(vehicle, ctx)
    assert _approx(ctx.int_step, 0.01)
    want_q1 = integrate(0.5, 0.0, 0.0, 0.01)
    assert store.get("q1") == want_q1
    assert _approx(store.get("q1d"), 0.5)


def test_out_step_fact_and_event_time():
    vehicle, kin, _ctx0 = _ready(sim_time=0.0)
    vehicle.store.set("out_step_fact", 0.5)
    ctx = _ctx(sim_time=1.5, event_time=0.2, int_step=0.001)
    kin.execute(vehicle, ctx)
    assert _approx(ctx.out_fact, 0.5)
    assert _approx(vehicle.store.get("event_time"), 0.2)
    assert _approx(vehicle.store.get("launch_time"), 1.5)


def test_phip_uses_small_not_eps_when_vbeb2_zero():
    vbeb = (16.0, 0.0, 16.0 * math.tan(10.0 * RAD))
    vehicle, kin, ctx = _ready(vbeb=vbeb, sim_time=0.0)
    kin.execute(vehicle, ctx)
    want = math.atan2(SMALL, vbeb[2])
    assert vehicle.store.get("phip") == pytest.approx(want, rel=RTOL, abs=ATOL)


def test_terminate_is_pass():
    vehicle, kin, ctx = _ready(sim_time=0.0)
    assert kin.terminate(vehicle, ctx) is None

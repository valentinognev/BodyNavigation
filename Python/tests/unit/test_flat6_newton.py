import math
from types import SimpleNamespace

import numpy as np

from cadac.constants import DEG, RAD
from cadac.env.gravity import gravity
from cadac.eom.flat6 import Flat6Newton
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat2tr, mat3tr

_ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
_VMASS = 9298.0
_DT = 0.001
_HBE0 = 1000.0
_DVBE0 = 180.0


def _vehicle():
    store = StateStore()
    vehicle = SimpleNamespace(store=store)
    newton = Flat6Newton()
    newton.define(vehicle)
    return vehicle, newton


def _plant(
    store,
    tbl=None,
    grav=None,
    wbeb=(0.0, 0.0, 0.0),
    fapb=(0.0, 0.0, 0.0),
    vmass=_VMASS,
):
    if tbl is None:
        tbl = np.eye(3)
    if grav is None:
        grav = gravity(_HBE0)
    if "TBL" not in store.names():
        store.define(Field("TBL", _ZEROS33, "mat", "out", "kinematics"))
    if "grav" not in store.names():
        store.define(Field("grav", 0.0, "real", "out", "environment"))
    if "WBEB" not in store.names():
        store.define(Field("WBEB", (0.0, 0.0, 0.0), "vec", "state", "euler"))
    if "FAPB" not in store.names():
        store.define(Field("FAPB", (0.0, 0.0, 0.0), "vec", "out", "forces"))
    if "vmass" not in store.names():
        store.define(Field("vmass", 0.0, "real", "init", "aerodynamics"))
    store.set("TBL", np.asarray(tbl, dtype=float))
    store.set("grav", float(grav))
    store.set("WBEB", np.asarray(wbeb, dtype=float))
    store.set("FAPB", np.asarray(fapb, dtype=float))
    store.set("vmass", float(vmass))


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


def _vbeb(alpha0x, beta0x, dvbe):
    salp = math.sin(alpha0x * RAD)
    calp = math.cos(alpha0x * RAD)
    sbet = math.sin(beta0x * RAD)
    cbet = math.cos(beta0x * RAD)
    return np.array(
        [calp * cbet * dvbe, sbet * dvbe, salp * cbet * dvbe], dtype=float
    )


def _flight_path(vbel):
    vbel1 = float(vbel[0])
    vbel2 = float(vbel[1])
    vbel3 = float(vbel[2])
    if vbel1 == 0.0 and vbel2 == 0.0:
        psivl = 0.0
    else:
        psivl = math.atan2(vbel2, vbel1)
    thtvl = math.atan2(-vbel3, math.sqrt(vbel1 * vbel1 + vbel2 * vbel2))
    return psivl, thtvl


def _cpp_newton_step(
    vbeb,
    vbebd,
    sbel,
    sbeld,
    sbelm,
    groundrange,
    tbl,
    wbeb,
    fapb,
    vmass,
    grav,
    dt,
    mfreeze=None,
    mfreeze_newt=0,
    dvbef=0.0,
):
    atb = _skew(wbeb) @ vbeb
    gravl = np.array([0.0, 0.0, grav])
    fspb = fapb * (1.0 / vmass)
    vbebd_new = fspb - atb + tbl @ gravl
    vbeb = integrate(vbebd_new, vbebd, vbeb, dt)
    vbel = tbl.T @ vbeb
    sbeld_new = vbel.copy()
    sbel = integrate(sbeld_new, sbeld, sbel, dt)
    psivl, thtvl = _flight_path(vbel)
    dvbe = float(np.linalg.norm(vbel))
    hbe = -float(sbel[2])
    anx = -fspb[2] / grav
    ayx = fspb[1] / grav
    tvl = mat2tr(psivl, thtvl)
    tvb = tvl @ tbl.T
    fspv = tvb @ fspb
    alx = fspv[1] / grav
    if mfreeze is not None:
        if mfreeze == 0:
            mfreeze_newt = 0
        else:
            if mfreeze != mfreeze_newt:
                mfreeze_newt = mfreeze
                dvbef = dvbe
            dvbe = dvbef
    del_sbel = sbel - sbelm
    del_sbel = del_sbel.copy()
    del_sbel[2] = 0.0
    groundrange = groundrange + float(np.linalg.norm(del_sbel))
    sbelm = sbel.copy()
    return {
        "VBEB": vbeb,
        "VBEBD": vbebd_new,
        "SBEL": sbel,
        "SBELD": sbeld_new,
        "VBEL": vbel,
        "FSPB": fspb,
        "ATB": atb,
        "dvbe": dvbe,
        "hbe": hbe,
        "psivlx": psivl * DEG,
        "thtvlx": thtvl * DEG,
        "anx": anx,
        "ayx": ayx,
        "alx": alx,
        "SBELM": sbelm,
        "groundrange": groundrange,
        "mfreeze_newt": mfreeze_newt,
        "dvbef": dvbef,
    }


def test_name_is_newton():
    assert Flat6Newton().name == "newton"


def test_define_registers_time_and_translation_state():
    vehicle, _newton = _vehicle()
    store = vehicle.store
    assert store.get("time") == 0.0
    assert store.field("time").role == "exec"
    assert store.field("time").outputs == ("scrn", "plot", "com")
    np.testing.assert_array_equal(store.get("VBEB"), np.zeros(3))
    np.testing.assert_array_equal(store.get("VBEBD"), np.zeros(3))
    np.testing.assert_array_equal(store.get("SBEL"), np.zeros(3))
    np.testing.assert_array_equal(store.get("SBELD"), np.zeros(3))
    np.testing.assert_array_equal(store.get("FSPB"), np.zeros(3))
    assert store.get("dvbe") == 0.0
    assert store.get("hbe") == 0.0
    assert store.get("sbel3") == 0.0
    assert store.get("alpha0x") == 0.0
    assert store.get("beta0x") == 0.0
    assert store.get("halt") == 0
    assert store.field("VBEB").outputs == ("plot",)
    assert store.field("SBEL").outputs == ("plot", "com")
    assert store.field("hbe").outputs == ("scrn", "plot")
    assert store.field("dvbe").outputs == ("plot",)


def test_define_does_not_register_other_module_fields():
    store = StateStore()
    Flat6Newton().define(SimpleNamespace(store=store))
    for name in ("TBL", "grav", "WBEB", "FAPB", "vmass", "mfreeze", "alphax"):
        assert name not in store.names()


def test_init_hbe_from_sbel3_then_one_step_stays_near():
    vehicle, newton = _vehicle()
    store = vehicle.store
    store.set("sbel3", -_HBE0)
    store.set("dvbe", _DVBE0)
    _plant(store, tbl=np.eye(3), grav=gravity(_HBE0), fapb=(0.0, 0.0, 0.0), vmass=_VMASS)
    newton.initialize(vehicle, None)
    assert store.get("hbe") == _HBE0
    np.testing.assert_array_equal(store.get("SBEL"), np.array([0.0, 0.0, -_HBE0]))
    newton.execute(vehicle, SimpleNamespace(sim_time=0.0, int_step=_DT))
    hbe = store.get("hbe")
    assert np.isfinite(hbe)
    np.testing.assert_allclose(hbe, _HBE0, atol=1e-3)
    assert abs(hbe) < 1e6


def test_initialize_vbeb_from_alpha0x_beta0x_degrees():
    vehicle, newton = _vehicle()
    store = vehicle.store
    store.set("sbel3", -_HBE0)
    store.set("dvbe", _DVBE0)
    store.set("alpha0x", 1.0)
    store.set("beta0x", 5.0)
    _plant(store, tbl=np.eye(3))
    newton.initialize(vehicle, None)
    np.testing.assert_allclose(
        store.get("VBEB"),
        _vbeb(1.0, 5.0, _DVBE0),
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_array_equal(store.get("VBEBD"), np.zeros(3))
    np.testing.assert_array_equal(store.get("SBELD"), np.zeros(3))


def test_initialize_vbel_uses_tbl_transpose_not_tbl():
    vehicle, newton = _vehicle()
    store = vehicle.store
    store.set("dvbe", _DVBE0)
    store.set("sbel3", -_HBE0)
    tbl = mat3tr(math.pi / 2.0, 0.0, 0.0)
    _plant(store, tbl=tbl)
    newton.initialize(vehicle, None)
    vbeb = _vbeb(0.0, 0.0, _DVBE0)
    np.testing.assert_allclose(store.get("VBEL"), tbl.T @ vbeb, rtol=1e-12, atol=1e-14)
    wrong = tbl @ vbeb
    assert abs(store.get("VBEL")[1] - wrong[1]) > 100.0
    np.testing.assert_allclose(store.get("psivlx"), 90.0, atol=1e-10)
    np.testing.assert_allclose(store.get("thtvlx"), 0.0, atol=1e-12)


def test_initialize_psivl_zero_when_horizontal_vbel_zero():
    vehicle, newton = _vehicle()
    store = vehicle.store
    store.set("dvbe", _DVBE0)
    store.set("sbel3", -_HBE0)
    tbl = mat3tr(0.0, math.pi / 2.0, 0.0)
    _plant(store, tbl=tbl)
    newton.initialize(vehicle, None)
    np.testing.assert_allclose(
        store.get("VBEL"),
        np.array([0.0, 0.0, -_DVBE0]),
        atol=1e-12,
    )
    assert store.get("psivlx") == 0.0
    np.testing.assert_allclose(store.get("thtvlx"), 90.0, atol=1e-12)
    np.testing.assert_array_equal(store.get("SBELM"), store.get("SBEL"))


def test_execute_sets_time_from_sim_time():
    vehicle, newton = _vehicle()
    store = vehicle.store
    store.set("sbel3", -_HBE0)
    store.set("dvbe", _DVBE0)
    _plant(store)
    newton.initialize(vehicle, None)
    assert store.get("time") == 0.0
    newton.execute(vehicle, SimpleNamespace(sim_time=1.5, int_step=_DT))
    assert store.get("time") == 1.5


def test_zero_fapb_identity_one_step_matches_cpp_replica():
    vehicle, newton = _vehicle()
    store = vehicle.store
    store.set("sbel3", -_HBE0)
    store.set("dvbe", _DVBE0)
    store.set("alpha0x", 1.0)
    grav = gravity(_HBE0)
    _plant(store, tbl=np.eye(3), grav=grav, fapb=(0.0, 0.0, 0.0), vmass=_VMASS)
    newton.initialize(vehicle, None)
    snap = _snapshot(store)
    newton.execute(vehicle, SimpleNamespace(sim_time=0.0, int_step=_DT))
    want = _cpp_newton_step(
        snap["VBEB"],
        snap["VBEBD"],
        snap["SBEL"],
        snap["SBELD"],
        snap["SBELM"],
        snap["groundrange"],
        snap["TBL"],
        snap["WBEB"],
        snap["FAPB"],
        snap["vmass"],
        snap["grav"],
        _DT,
    )
    _assert_step(store, want)


def test_execute_second_step_uses_stored_slopes():
    vehicle, newton = _vehicle()
    store = vehicle.store
    store.set("sbel3", -_HBE0)
    store.set("dvbe", _DVBE0)
    store.set("alpha0x", 1.0)
    fapb = (1000.0, -200.0, 500.0)
    wbeb = (0.1, -0.05, 0.02)
    grav = gravity(_HBE0)
    _plant(store, tbl=np.eye(3), grav=grav, wbeb=wbeb, fapb=fapb, vmass=_VMASS)
    newton.initialize(vehicle, None)
    ctx = SimpleNamespace(sim_time=0.0, int_step=_DT)
    newton.execute(vehicle, ctx)
    snap = _snapshot(store)
    newton.execute(vehicle, SimpleNamespace(sim_time=_DT, int_step=_DT))
    want = _cpp_newton_step(
        snap["VBEB"],
        snap["VBEBD"],
        snap["SBEL"],
        snap["SBELD"],
        snap["SBELM"],
        snap["groundrange"],
        snap["TBL"],
        snap["WBEB"],
        snap["FAPB"],
        snap["vmass"],
        snap["grav"],
        _DT,
    )
    _assert_step(store, want)
    assert store.get("time") == _DT


def test_atb_from_skew_wbeb():
    vehicle, newton = _vehicle()
    store = vehicle.store
    store.set("sbel3", -_HBE0)
    store.set("dvbe", _DVBE0)
    _plant(store, tbl=np.eye(3), wbeb=(0.0, 0.0, 1.0), fapb=(0.0, 0.0, 0.0))
    newton.initialize(vehicle, None)
    newton.execute(vehicle, SimpleNamespace(sim_time=0.0, int_step=_DT))
    np.testing.assert_allclose(store.get("ATB"), np.array([0.0, _DVBE0, 0.0]), atol=1e-12)


def test_anx_ayx_alx_from_fspb_and_mat2tr():
    vehicle, newton = _vehicle()
    store = vehicle.store
    store.set("sbel3", -_HBE0)
    store.set("dvbe", _DVBE0)
    grav = gravity(_HBE0)
    fapb = (0.0, 2.0 * _VMASS * grav, -3.0 * _VMASS * grav)
    _plant(store, tbl=np.eye(3), grav=grav, fapb=fapb, vmass=_VMASS)
    newton.initialize(vehicle, None)
    newton.execute(vehicle, SimpleNamespace(sim_time=0.0, int_step=_DT))
    np.testing.assert_allclose(
        store.get("FSPB"),
        np.array(fapb, dtype=float) * (1.0 / _VMASS),
        rtol=1e-12,
    )
    np.testing.assert_allclose(store.get("anx"), 3.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(store.get("ayx"), 2.0, rtol=1e-12, atol=1e-12)
    fspb = store.get("FSPB")
    psivl, thtvl = _flight_path(store.get("VBEL"))
    tvl = mat2tr(psivl, thtvl)
    fspv = (tvl @ np.eye(3).T) @ fspb
    np.testing.assert_allclose(store.get("alx"), fspv[1] / grav, rtol=1e-12, atol=1e-12)


def test_groundrange_zeros_vertical_component():
    vehicle, newton = _vehicle()
    store = vehicle.store
    store.set("sbel3", -_HBE0)
    store.set("dvbe", _DVBE0)
    store.set("alpha0x", 90.0)
    _plant(store, tbl=np.eye(3), fapb=(0.0, 0.0, 0.0))
    newton.initialize(vehicle, None)
    sbel_old = store.get("SBEL").copy()
    newton.execute(vehicle, SimpleNamespace(sim_time=0.0, int_step=_DT))
    vertical = abs(store.get("SBEL")[2] - sbel_old[2])
    assert vertical > 1e-8
    np.testing.assert_allclose(store.get("groundrange"), 0.0, atol=1e-12)
    assert store.get("groundrange") < 0.01 * vertical


def test_groundrange_horizontal_increment():
    vehicle, newton = _vehicle()
    store = vehicle.store
    store.set("sbel3", -_HBE0)
    store.set("dvbe", _DVBE0)
    _plant(store, tbl=np.eye(3), fapb=(0.0, 0.0, 0.0))
    newton.initialize(vehicle, None)
    sbel_old = store.get("SBEL").copy()
    newton.execute(vehicle, SimpleNamespace(sim_time=0.0, int_step=_DT))
    delta = store.get("SBEL") - sbel_old
    delta[2] = 0.0
    np.testing.assert_allclose(
        store.get("groundrange"),
        float(np.linalg.norm(delta)),
        rtol=1e-12,
        atol=1e-14,
    )
    assert store.get("groundrange") > 0.0


def test_skip_mfreeze_when_not_on_store():
    vehicle, newton = _vehicle()
    store = vehicle.store
    store.set("sbel3", -_HBE0)
    store.set("dvbe", _DVBE0)
    _plant(store)
    newton.initialize(vehicle, None)
    newton.execute(vehicle, SimpleNamespace(sim_time=0.0, int_step=_DT))
    assert "mfreeze" not in store.names()
    assert np.isfinite(store.get("dvbe"))
    assert store.get("dvbe") > 0.0


def test_mfreeze_latches_dvbe_when_present():
    vehicle, newton = _vehicle()
    store = vehicle.store
    store.set("sbel3", -_HBE0)
    store.set("dvbe", _DVBE0)
    _plant(store, fapb=(0.0, 0.0, 0.0))
    store.define(Field("mfreeze", 0, "int", "data", "control"))
    store.set("mfreeze", 1)
    newton.initialize(vehicle, None)
    newton.execute(vehicle, SimpleNamespace(sim_time=0.0, int_step=_DT))
    live = store.get("dvbe")
    assert store.get("mfreeze_newt") == 1
    np.testing.assert_allclose(store.get("dvbef"), live, rtol=1e-12, atol=1e-14)
    store.set("FAPB", np.array([1.0e6, 0.0, 0.0]))
    newton.execute(vehicle, SimpleNamespace(sim_time=_DT, int_step=_DT))
    np.testing.assert_allclose(store.get("dvbe"), live, rtol=1e-12, atol=1e-14)
    assert store.get("mfreeze_newt") == 1
    np.testing.assert_allclose(store.get("dvbef"), live, rtol=1e-12, atol=1e-14)


def _snapshot(store):
    return {
        "VBEB": store.get("VBEB").copy(),
        "VBEBD": store.get("VBEBD").copy(),
        "SBEL": store.get("SBEL").copy(),
        "SBELD": store.get("SBELD").copy(),
        "SBELM": store.get("SBELM").copy(),
        "groundrange": store.get("groundrange"),
        "TBL": store.get("TBL").copy(),
        "WBEB": store.get("WBEB").copy(),
        "FAPB": store.get("FAPB").copy(),
        "vmass": store.get("vmass"),
        "grav": store.get("grav"),
    }


def _assert_step(store, want):
    for name in ("VBEB", "VBEBD", "SBEL", "SBELD", "VBEL", "FSPB", "ATB", "SBELM"):
        np.testing.assert_allclose(store.get(name), want[name], rtol=1e-12, atol=1e-14)
    for name in ("dvbe", "hbe", "psivlx", "thtvlx", "anx", "ayx", "alx", "groundrange"):
        np.testing.assert_allclose(store.get(name), want[name], rtol=1e-12, atol=1e-14)

from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import DEG, RAD, WEII3
from cadac.eom.round6 import Round6Newton
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat3tr, polar_from_cart
from cadac.math.wgs84 import cad_geo84_in, cad_in_geo84, cad_tdi84, cad_tgi84

_ZEROS3 = (0.0, 0.0, 0.0)
_ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
_RTOL = 1e-12
_ATOL = 1e-14


def _ctx(sim_time=0.0, int_step=0.01):
    return SimContext(
        sim_time=sim_time,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _vehicle():
    store = StateStore()
    vehicle = SimpleNamespace(store=store)
    newton = Round6Newton()
    newton.define(vehicle)
    return vehicle, newton


def _plant_kinematics(
    store,
    *,
    time=0.0,
    psibdx=0.0,
    thtbdx=2.5,
    phibdx=0.0,
):
    for name, value, role in (
        ("time", time, "exec"),
        ("psibdx", psibdx, "in/di"),
        ("thtbdx", thtbdx, "in/di"),
        ("phibdx", phibdx, "in/di"),
    ):
        if name not in store.names():
            store.define(Field(name, 0.0, "real", role, "kinematics"))
        store.set(name, value)


def _climb_ics(store, minit=0):
    store.set("minit", minit)
    store.set("lonx", 10.0)
    store.set("latx", 10.0)
    store.set("alt", 10000.0)
    store.set("dvbe", 1000.0)
    store.set("alpha0x", 2.5)
    store.set("beta0x", 0.0)
    _plant_kinematics(store)


def _vbeb(alpha0x, beta0x, dvbe):
    salp = np.sin(alpha0x * RAD)
    calp = np.cos(alpha0x * RAD)
    sbet = np.sin(beta0x * RAD)
    cbet = np.cos(beta0x * RAD)
    return np.array(
        [calp * cbet * dvbe, sbet * dvbe, salp * cbet * dvbe], dtype=float
    )


def _weii():
    weii = np.zeros((3, 3))
    weii[0, 1] = -WEII3
    weii[1, 0] = WEII3
    return weii


def _cpp_init_newton(lonx, latx, alt, dvbe, alpha0x, beta0x, psibdx, thtbdx, phibdx, time):
    sbii = cad_in_geo84(lonx * RAD, latx * RAD, alt, time)
    dbi = float(np.linalg.norm(sbii))
    vbeb = _vbeb(alpha0x, beta0x, dvbe)
    tbd = mat3tr(psibdx * RAD, thtbdx * RAD, phibdx * RAD)
    vbed = tbd.T @ vbeb
    tdi = cad_tdi84(lonx * RAD, latx * RAD, alt, time)
    tgi = cad_tgi84(lonx * RAD, latx * RAD, alt, time)
    weii = _weii()
    vbii = tdi.T @ vbed + weii @ sbii
    dvbi = float(np.linalg.norm(vbii))
    polar = polar_from_cart(vbed)
    return {
        "SBII": sbii,
        "VBED": vbed,
        "VBII": vbii,
        "TDI": tdi,
        "TGI": tgi,
        "WEII": weii,
        "dbi": dbi,
        "dvbi": dvbi,
        "psivdx": DEG * float(polar[1]),
        "thtvdx": DEG * float(polar[2]),
    }


def test_name_is_newton():
    assert Round6Newton().name == "newton"


def test_define_registers_def_newton_fields():
    store = StateStore()
    Round6Newton().define(SimpleNamespace(store=store))
    for name in (
        "minit",
        "alpha0x",
        "beta0x",
        "lonx",
        "latx",
        "alt",
        "TVD",
        "TDI",
        "dvbe",
        "dvbi",
        "WEII",
        "psivdx",
        "thtvdx",
        "dbi",
        "TGI",
        "VBED",
        "altx",
        "SBII",
        "VBII",
        "ABII",
        "grndtrck",
        "FSPB",
        "ayx",
        "anx",
        "gndtrkmx",
        "gndtrnmx",
        "latx_bias",
        "dvbi_bias",
        "dbi_bias",
        "mfreeze_newt",
        "dvbef",
        "thtvdx_bias",
        "sat_semi",
        "sat_ecc",
        "sat_inclx",
        "sat_lon_anodex",
        "sat_arg_perix",
        "sat_true_anomx",
        "ranglex_l_t",
        "headon_flag",
        "tgo_insertion",
    ):
        assert name in store.names()
    assert store.get("minit") == 0
    assert store.get("headon_flag") == 0
    assert store.get("mfreeze_newt") == 0
    assert store.get("lonx") == 0.0
    assert store.get("latx") == 0.0
    assert store.get("alt") == 0.0
    assert store.get("dvbe") == 0.0
    assert store.get("alpha0x") == 0.0
    assert store.get("beta0x") == 0.0
    np.testing.assert_array_equal(store.get("SBII"), np.zeros(3))
    np.testing.assert_array_equal(store.get("VBII"), np.zeros(3))
    np.testing.assert_array_equal(store.get("ABII"), np.zeros(3))
    np.testing.assert_array_equal(store.get("VBED"), np.zeros(3))
    np.testing.assert_array_equal(store.get("FSPB"), np.zeros(3))
    np.testing.assert_array_equal(store.get("TDI"), np.zeros((3, 3)))
    np.testing.assert_array_equal(store.get("TGI"), np.zeros((3, 3)))
    np.testing.assert_array_equal(store.get("WEII"), np.zeros((3, 3)))
    np.testing.assert_array_equal(store.get("TVD"), np.zeros((3, 3)))
    assert store.field("minit").role == "data"
    assert store.field("minit").type == "int"
    assert store.field("lonx").role == "init/diag"
    assert store.field("lonx").outputs == ("scrn", "plot", "com")
    assert store.field("latx").role == "init/diag"
    assert store.field("alt").role == "init/out"
    assert store.field("dvbe").role == "init/out"
    assert store.field("alpha0x").role == "data"
    assert store.field("beta0x").role == "data"
    assert store.field("TDI").role == "init"
    assert store.field("WEII").role == "init"
    assert store.field("TGI").role == "init"
    assert store.field("TVD").role == "out"
    assert store.field("SBII").role == "state"
    assert store.field("SBII").outputs == ("com",)
    assert store.field("VBII").role == "state"
    assert store.field("VBII").outputs == ("com",)
    assert store.field("ABII").role == "save"
    assert store.field("FSPB").role == "out"
    assert store.field("VBED").role == "out"
    assert store.field("psivdx").role == "init/out"
    assert store.field("thtvdx").role == "init/out"
    assert store.field("dvbi").outputs == ("scrn", "plot", "com")
    assert store.field("dbi").outputs == ("scrn", "plot")
    assert store.field("ayx").outputs == ("plot",)
    assert store.field("anx").outputs == ("plot",)


def test_define_does_not_register_kinematics_euler_environment_or_hyper():
    store = StateStore()
    Round6Newton().define(SimpleNamespace(store=store))
    for name in (
        "time",
        "TBI",
        "TBD",
        "GRAVG",
        "psibdx",
        "thtbdx",
        "phibdx",
        "WBIB",
        "FAPB",
        "vmass",
        "mfreeze",
        "wp_lonx",
        "wp_latx",
        "dbi_desired",
        "dvbi_desired",
        "thtvdx_desired",
        "VBEB",
        "ABEL",
    ):
        assert name not in store.names()


def test_initialize_climb_ics_geo84_alt_and_dvbe():
    vehicle, newton = _vehicle()
    _climb_ics(vehicle.store)
    newton.initialize(vehicle, _ctx())
    store = vehicle.store
    lon, lat, alt = cad_geo84_in(store.get("SBII"), store.get("time"))
    np.testing.assert_allclose(alt, 10000.0, atol=1e-6)
    np.testing.assert_allclose([lon, lat], [10.0 * RAD, 10.0 * RAD], atol=1e-6)
    np.testing.assert_allclose(store.get("dvbe"), 1000.0, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("alt"), 10000.0, rtol=_RTOL, atol=_ATOL)


def test_initialize_sbii_matches_cad_in_geo84():
    vehicle, newton = _vehicle()
    _climb_ics(vehicle.store)
    newton.initialize(vehicle, _ctx())
    expected = cad_in_geo84(10.0 * RAD, 10.0 * RAD, 10000.0, 0.0)
    np.testing.assert_allclose(
        vehicle.store.get("SBII"), expected, rtol=_RTOL, atol=_ATOL
    )


def test_initialize_vbed_from_alpha_beta_dvbe():
    vehicle, newton = _vehicle()
    _climb_ics(vehicle.store)
    newton.initialize(vehicle, _ctx())
    want = _cpp_init_newton(10.0, 10.0, 10000.0, 1000.0, 2.5, 0.0, 0.0, 2.5, 0.0, 0.0)
    np.testing.assert_allclose(
        vehicle.store.get("VBED"), want["VBED"], rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        np.linalg.norm(vehicle.store.get("VBED")), 1000.0, rtol=_RTOL, atol=_ATOL
    )


def test_initialize_vbii_from_tdi_weii():
    vehicle, newton = _vehicle()
    _climb_ics(vehicle.store)
    newton.initialize(vehicle, _ctx())
    want = _cpp_init_newton(10.0, 10.0, 10000.0, 1000.0, 2.5, 0.0, 0.0, 2.5, 0.0, 0.0)
    np.testing.assert_allclose(
        vehicle.store.get("VBII"), want["VBII"], rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("TDI"), want["TDI"], rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("TGI"), want["TGI"], rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("WEII"), want["WEII"], rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("dvbi"), want["dvbi"], rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("dbi"), want["dbi"], rtol=_RTOL, atol=_ATOL
    )


def test_initialize_flight_path_angles_in_degrees():
    vehicle, newton = _vehicle()
    _climb_ics(vehicle.store)
    newton.initialize(vehicle, _ctx())
    want = _cpp_init_newton(10.0, 10.0, 10000.0, 1000.0, 2.5, 0.0, 0.0, 2.5, 0.0, 0.0)
    np.testing.assert_allclose(
        vehicle.store.get("psivdx"), want["psivdx"], rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("thtvdx"), want["thtvdx"], rtol=_RTOL, atol=_ATOL
    )
    assert abs(vehicle.store.get("psivdx")) < 180.0
    assert abs(vehicle.store.get("thtvdx")) < 90.0


def test_initialize_abii_is_zeros():
    vehicle, newton = _vehicle()
    _climb_ics(vehicle.store)
    newton.initialize(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("ABII"), np.zeros(3))


def test_initialize_does_not_write_fspb():
    vehicle, newton = _vehicle()
    _climb_ics(vehicle.store)
    newton.initialize(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("FSPB"), np.zeros(3))


def test_initialize_unknown_minit_raises():
    for minit in (2, 99, -1):
        vehicle, newton = _vehicle()
        _climb_ics(vehicle.store, minit=minit)
        with pytest.raises(ValueError, match="unknown minit"):
            newton.initialize(vehicle, _ctx())


def test_initialize_minit1_with_elements():
    vehicle, newton = _vehicle()
    store = vehicle.store
    for name in (
        "dbi_desired",
        "dvbi_desired",
        "thtvdx_desired",
        "wp_lonx",
        "wp_latx",
    ):
        store.define(Field(name, 0.0, "real", "data", "guidance"))
    _climb_ics(store, minit=1)
    semi = 7_000_000.0
    store.set("sat_semi", semi)
    store.set("sat_ecc", 0.0)
    store.set("sat_inclx", 0.0)
    store.set("sat_lon_anodex", 0.0)
    store.set("sat_arg_perix", 0.0)
    store.set("sat_true_anomx", 0.0)
    store.set("ranglex_l_t", 0.0)
    store.set("headon_flag", 0)
    store.set("tgo_insertion", 0.0)
    newton.initialize(vehicle, _ctx())
    np.testing.assert_allclose(store.get("lonx"), 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(store.get("latx"), 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(store.get("psibdx"), 90.0, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(store.get("dbi_desired"), semi, rtol=1e-12, atol=0.0)


def test_initialize_nonzero_beta_and_heading_matches_cpp_replica():
    vehicle, newton = _vehicle()
    _climb_ics(vehicle.store)
    store = vehicle.store
    store.set("alpha0x", 5.0)
    store.set("beta0x", -2.0)
    store.set("psibdx", 30.0)
    store.set("thtbdx", 4.0)
    store.set("phibdx", 1.0)
    newton.initialize(vehicle, _ctx())
    want = _cpp_init_newton(10.0, 10.0, 10000.0, 1000.0, 5.0, -2.0, 30.0, 4.0, 1.0, 0.0)
    np.testing.assert_allclose(store.get("VBED"), want["VBED"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("VBII"), want["VBII"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("psivdx"), want["psivdx"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("thtvdx"), want["thtvdx"], rtol=_RTOL, atol=_ATOL)


def test_terminate_exists_and_is_pass():
    vehicle, newton = _vehicle()
    _climb_ics(vehicle.store)
    newton.initialize(vehicle, _ctx())
    before = vehicle.store.get("SBII").copy()
    newton.terminate(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("SBII"), before)

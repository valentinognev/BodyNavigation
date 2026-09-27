from math import atan2, cos, sin, sqrt
from pathlib import Path

import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.sam6.newton import Sam6Newton

RTOL = 1e-12
ATOL = 1e-14

DEFINED = (
    "VBEBD",
    "VBEB",
    "SBELD",
    "SBEL",
    "sbel1",
    "sbel2",
    "sbel3",
    "FSPB",
    "VBEL",
    "dvbe",
    "alpha0x",
    "beta0x",
    "alt",
    "hbe",
    "psivlx",
    "thtvlx",
    "anx",
    "ayx",
    "ATB",
    "mfreeze_newt",
    "dvbef",
    "SLEL",
)
ROLES = {
    "VBEBD": "state",
    "VBEB": "state",
    "SBELD": "state",
    "SBEL": "state",
    "sbel1": "data",
    "sbel2": "data",
    "sbel3": "data",
    "FSPB": "out",
    "VBEL": "out",
    "dvbe": "in/out",
    "alpha0x": "data",
    "beta0x": "data",
    "alt": "out",
    "hbe": "out",
    "psivlx": "diag",
    "thtvlx": "diag",
    "anx": "diag",
    "ayx": "diag",
    "ATB": "diag",
    "mfreeze_newt": "save",
    "dvbef": "save",
    "SLEL": "out",
}
OUTPUTS = {
    "VBEBD": (),
    "VBEB": (),
    "SBELD": (),
    "SBEL": ("scrn", "plot", "com"),
    "sbel1": (),
    "sbel2": (),
    "sbel3": (),
    "FSPB": (),
    "VBEL": ("scrn", "com"),
    "dvbe": ("scrn", "plot"),
    "alpha0x": (),
    "beta0x": (),
    "alt": ("scrn", "plot", "com"),
    "hbe": ("scrn", "plot"),
    "psivlx": ("scrn", "plot"),
    "thtvlx": ("scrn", "plot"),
    "anx": ("scrn", "plot"),
    "ayx": ("scrn", "plot"),
    "ATB": (),
    "mfreeze_newt": (),
    "dvbef": (),
    "SLEL": (),
}
VEC_FIELDS = ("VBEBD", "VBEB", "SBELD", "SBEL", "FSPB", "VBEL", "ATB", "SLEL")
INT_FIELDS = ("mfreeze_newt",)
NOT_DEFINED = (
    "mass",
    "vmass",
    "FAPB",
    "TBL",
    "WBEB",
    "grav",
    "mfreeze",
    "time",
    "halt",
    "SBELM",
    "groundrange",
    "alx",
    "FMB",
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


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


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


def _vbeb0(dvbe, alpha0x, beta0x):
    salp = sin(alpha0x * RAD)
    calp = cos(alpha0x * RAD)
    sbet = sin(beta0x * RAD)
    cbet = cos(beta0x * RAD)
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
        psivl = atan2(vbel2, vbel1)
    thtvl = atan2(-vbel3, sqrt(vbel1 * vbel1 + vbel2 * vbel2))
    return psivl * DEG, thtvl * DEG


def _cpp_step(fapb, mass, wbeb, tbl, grav, vbebd, vbeb, sbeld, sbel, dt):
    atb = _skew(wbeb) @ vbeb
    gravl = np.array([0.0, 0.0, grav], dtype=float)
    fspb = fapb * (1.0 / mass)
    vbebd_new = fspb - atb + tbl @ gravl
    vbeb = integrate(vbebd_new, vbebd, vbeb, dt)
    vbebd = vbebd_new
    vbel = tbl.T @ vbeb
    sbeld_new = vbel
    sbel = integrate(sbeld_new, sbeld, sbel, dt)
    sbeld = sbeld_new
    dvbe = float(np.linalg.norm(vbel))
    alt = -float(sbel[2])
    hbe = alt
    anx = -float(fspb[2]) / grav
    ayx = float(fspb[1]) / grav
    psivlx, thtvlx = _flight_path(vbel)
    return {
        "VBEBD": vbebd,
        "VBEB": vbeb,
        "SBELD": sbeld,
        "SBEL": sbel,
        "VBEL": vbel,
        "FSPB": fspb,
        "ATB": atb,
        "dvbe": dvbe,
        "alt": alt,
        "hbe": hbe,
        "anx": anx,
        "ayx": ayx,
        "psivlx": psivlx,
        "thtvlx": thtvlx,
    }


def _plant_tbl(store, tbl):
    store.define(Field("TBL", tbl, "mat", "out", "kinematics"))


def _plant_exec(store, *, mass, fapb, grav, wbeb):
    store.define(Field("mass", mass, "real", "out", "propulsion"))
    store.define(Field("FAPB", fapb, "vec", "out", "forces"))
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("WBEB", wbeb, "vec", "diag", "euler"))


def _defined(*, dvbe=16.0, alpha0x=0.0, beta0x=0.0, sbel=(0.0, 0.0, 0.0), tbl=None):
    vehicle = _Vehicle()
    newton = Sam6Newton()
    newton.define(vehicle)
    if tbl is None:
        tbl = np.eye(3)
    _plant_tbl(vehicle.store, tbl)
    vehicle.store.set("dvbe", dvbe)
    vehicle.store.set("alpha0x", alpha0x)
    vehicle.store.set("beta0x", beta0x)
    vehicle.store.set("sbel1", sbel[0])
    vehicle.store.set("sbel2", sbel[1])
    vehicle.store.set("sbel3", sbel[2])
    return vehicle, newton


def _ready(
    *,
    dvbe=16.0,
    alpha0x=0.0,
    beta0x=0.0,
    sbel=(0.0, 0.0, 0.0),
    tbl=None,
    mass=300.0,
    fapb=(0.0, 0.0, 0.0),
    grav=9.8,
    wbeb=(0.0, 0.0, 0.0),
):
    vehicle, newton = _defined(
        dvbe=dvbe, alpha0x=alpha0x, beta0x=beta0x, sbel=sbel, tbl=tbl
    )
    newton.initialize(vehicle, _ctx())
    _plant_exec(vehicle.store, mass=mass, fapb=fapb, grav=grav, wbeb=wbeb)
    return vehicle, newton


def test_name_is_newton():
    assert Sam6Newton().name == "newton"


def test_define_cpp_fields():
    vehicle = _Vehicle()
    Sam6Newton().define(vehicle)
    store = vehicle.store
    assert tuple(store.names()) == DEFINED
    for name in DEFINED:
        field = store.field(name)
        assert field.module == "newton"
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        if name in VEC_FIELDS:
            assert field.type == "vec"
            np.testing.assert_allclose(
                store.get(name), np.zeros(3), rtol=RTOL, atol=ATOL
            )
        elif name in INT_FIELDS:
            assert field.type == "int"
            assert store.get(name) == 0
        else:
            assert field.type == "real"
            assert _approx(store.get(name), 0.0), name


def test_define_does_not_register_mass_forces_or_flat6_extras():
    vehicle = _Vehicle()
    Sam6Newton().define(vehicle)
    for name in NOT_DEFINED:
        assert name not in vehicle.store.names()


def test_init_zero_sbel_identity_tbl_writes_alt_hbe_and_vbeb():
    vehicle, newton = _defined(dvbe=16.0, alpha0x=0.0, beta0x=0.0, sbel=(0.0, 0.0, 0.0))
    newton.initialize(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("alt"), 0.0)
    assert _approx(store.get("hbe"), 0.0)
    vbeb = store.get("VBEB")
    np.testing.assert_allclose(np.linalg.norm(vbeb), 16.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VBEL"), vbeb, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vbeb, [16.0, 0.0, 0.0], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBEL"), [0.0, 0.0, 0.0], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SLEL"), store.get("SBEL"), rtol=RTOL, atol=ATOL)


def test_init_vbel_is_tbl_transpose_times_vbeb():
    psi = 90.0 * RAD
    tbl = np.array(
        [
            [cos(psi), sin(psi), 0.0],
            [-sin(psi), cos(psi), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    vehicle, newton = _defined(dvbe=16.0, tbl=tbl)
    newton.initialize(vehicle, _ctx())
    vbeb = vehicle.store.get("VBEB")
    np.testing.assert_allclose(
        vehicle.store.get("VBEL"), tbl.T @ vbeb, rtol=RTOL, atol=ATOL
    )
    assert not np.allclose(vehicle.store.get("VBEL"), vbeb, rtol=RTOL, atol=ATOL)


def test_init_alt_and_hbe_are_minus_sbel_z():
    vehicle, newton = _defined(sbel=(10.0, 20.0, -1000.0))
    newton.initialize(vehicle, _ctx())
    store = vehicle.store
    np.testing.assert_allclose(store.get("SBEL"), [10.0, 20.0, -1000.0], rtol=RTOL, atol=ATOL)
    assert _approx(store.get("alt"), 1000.0)
    assert _approx(store.get("hbe"), 1000.0)
    np.testing.assert_allclose(store.get("SLEL"), store.get("SBEL"), rtol=RTOL, atol=ATOL)


def test_one_execute_alt_finite():
    vehicle, newton = _ready()
    newton.execute(vehicle, _ctx(0.001))
    alt = vehicle.store.get("alt")
    assert np.isfinite(alt)
    hbe = vehicle.store.get("hbe")
    assert np.isfinite(hbe)
    assert _approx(alt, -float(vehicle.store.get("SBEL")[2]))
    assert _approx(hbe, alt)


def test_fspb_is_fapb_over_mass_not_vmass():
    vehicle, newton = _ready(fapb=(300.0, 0.0, 0.0), mass=300.0)
    vehicle.store.define(Field("vmass", 1.0, "real", "out", "propulsion"))
    newton.execute(vehicle, _ctx(0.001))
    np.testing.assert_allclose(vehicle.store.get("FSPB"), [1.0, 0.0, 0.0], rtol=RTOL, atol=ATOL)


def test_one_step_matches_cpp_integrate():
    mass = 300.0
    grav = 9.8
    dt = 0.001
    fapb = np.array([0.0, 0.0, 0.0], dtype=float)
    wbeb = np.array([0.0, 0.0, 0.0], dtype=float)
    tbl = np.eye(3)
    vehicle, newton = _ready(mass=mass, fapb=fapb, grav=grav, wbeb=wbeb, tbl=tbl)
    newton.execute(vehicle, _ctx(dt))
    want = _cpp_step(
        fapb,
        mass,
        wbeb,
        tbl,
        grav,
        np.zeros(3),
        np.array([16.0, 0.0, 0.0], dtype=float),
        np.zeros(3),
        np.zeros(3),
        dt,
    )
    store = vehicle.store
    for name in (
        "VBEB",
        "VBEL",
        "SBEL",
        "FSPB",
        "ATB",
        "VBEBD",
        "SBELD",
    ):
        np.testing.assert_allclose(store.get(name), want[name], rtol=RTOL, atol=ATOL)
    assert _approx(store.get("dvbe"), want["dvbe"])
    assert _approx(store.get("alt"), want["alt"])
    assert _approx(store.get("hbe"), want["hbe"])
    assert _approx(store.get("anx"), want["anx"])
    assert _approx(store.get("ayx"), want["ayx"])
    assert _approx(store.get("psivlx"), want["psivlx"])
    assert _approx(store.get("thtvlx"), want["thtvlx"])


def test_second_step_uses_stored_vbebd():
    mass = 300.0
    grav = 9.8
    dt = 0.001
    fapb = np.zeros(3)
    wbeb = np.zeros(3)
    tbl = np.eye(3)
    vehicle, newton = _ready(mass=mass, fapb=fapb, grav=grav, wbeb=wbeb)
    ctx = _ctx(dt)
    newton.execute(vehicle, ctx)
    store = vehicle.store
    vbebd = np.array(store.get("VBEBD"), copy=True)
    vbeb = np.array(store.get("VBEB"), copy=True)
    sbeld = np.array(store.get("SBELD"), copy=True)
    sbel = np.array(store.get("SBEL"), copy=True)
    newton.execute(vehicle, ctx)
    want = _cpp_step(fapb, mass, wbeb, tbl, grav, vbebd, vbeb, sbeld, sbel, dt)
    np.testing.assert_allclose(store.get("VBEB"), want["VBEB"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("SBEL"), want["SBEL"], rtol=RTOL, atol=ATOL)
    fresh = integrate(vbebd, np.zeros(3), vbeb, dt)
    assert store.get("VBEB") != pytest.approx(fresh, rel=RTOL, abs=ATOL)


def test_tangential_accel_subtracts_atb():
    mass = 300.0
    grav = 9.8
    dt = 0.001
    fapb = np.zeros(3)
    wbeb = np.array([0.0, 0.1, 0.0], dtype=float)
    tbl = np.eye(3)
    vehicle, newton = _ready(mass=mass, fapb=fapb, grav=grav, wbeb=wbeb)
    newton.execute(vehicle, _ctx(dt))
    want = _cpp_step(
        fapb,
        mass,
        wbeb,
        tbl,
        grav,
        np.zeros(3),
        np.array([16.0, 0.0, 0.0], dtype=float),
        np.zeros(3),
        np.zeros(3),
        dt,
    )
    np.testing.assert_allclose(
        vehicle.store.get("ATB"), want["ATB"], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("VBEBD"), want["VBEBD"], rtol=RTOL, atol=ATOL
    )
    no_atb = fapb / mass + tbl @ np.array([0.0, 0.0, grav])
    assert vehicle.store.get("VBEBD") != pytest.approx(no_atb, rel=RTOL, abs=ATOL)


def test_skips_mfreeze_when_absent():
    vehicle, newton = _ready()
    assert "mfreeze" not in vehicle.store.names()
    newton.execute(vehicle, _ctx(0.001))
    dvbe_first = vehicle.store.get("dvbe")
    newton.execute(vehicle, _ctx(0.001))
    assert vehicle.store.get("dvbe") != pytest.approx(dvbe_first, rel=RTOL, abs=ATOL)


def test_mfreeze_latches_dvbe_when_present():
    vehicle, newton = _ready()
    vehicle.store.define(Field("mfreeze", 1, "int", "data", "control"))
    ctx = _ctx(0.001)
    newton.execute(vehicle, ctx)
    frozen = vehicle.store.get("dvbe")
    newton.execute(vehicle, ctx)
    assert _approx(vehicle.store.get("dvbe"), frozen)
    assert vehicle.store.get("mfreeze_newt") == 1
    assert _approx(vehicle.store.get("dvbef"), frozen)


def test_is_not_flat6_newton_subclass():
    from cadac.eom.flat6 import Flat6Newton

    assert not issubclass(Sam6Newton, Flat6Newton)


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.flat6.sam6.newton as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6Newton" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src
    assert "from cadac.kernel.integrate import integrate" in src
    assert "mat2tr" in src
    assert "RAD" in src
    assert "DEG" in src


def test_terminate_exists_and_is_pass():
    vehicle, newton = _ready()
    assert newton.terminate(vehicle, _ctx()) is None

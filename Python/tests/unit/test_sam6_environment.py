from pathlib import Path

import numpy as np
import pytest

from cadac.constants import EARTH_MASS, G, R, REARTH
from cadac.env.us76 import atmosphere76
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.sam6.environment import Sam6Environment
from cadac.vehicles.sam6.newton import Sam6Newton

RTOL = 1e-12
ATOL = 1e-14

DEFINED = (
    "press",
    "rho",
    "vsound",
    "grav",
    "vmach",
    "pdynmc",
    "tempk",
    "mfreeze_environ",
    "pdynmcf",
    "machf",
)
ROLES = {
    "press": "out",
    "rho": "out",
    "vsound": "diag",
    "grav": "out",
    "vmach": "out",
    "pdynmc": "out",
    "tempk": "out",
    "mfreeze_environ": "save",
    "pdynmcf": "save",
    "machf": "save",
}
OUTPUTS = {
    "press": (),
    "rho": (),
    "vsound": (),
    "grav": (),
    "vmach": ("scrn", "plot", "com"),
    "pdynmc": ("scrn", "plot", "com"),
    "tempk": (),
    "mfreeze_environ": (),
    "pdynmcf": (),
    "machf": (),
}
INT_FIELDS = ("mfreeze_environ",)
NOT_DEFINED = (
    "alt",
    "hbe",
    "dvbe",
    "VBEL",
    "SBEL",
    "mfreeze",
    "mguide",
    "trcond",
    "trdynm",
    "mwind",
    "VAEL",
    "VBAL",
    "dvba",
    "mach",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.001,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _grav(alt):
    return G * EARTH_MASS / (REARTH + alt) ** 2


def _newton_init_then_env(
    *,
    dvbe=16.0,
    sbel=(0.0, 0.0, -1000.0),
    alpha0x=0.0,
    beta0x=0.0,
):
    vehicle = _Vehicle()
    newton = Sam6Newton()
    env = Sam6Environment()
    newton.define(vehicle)
    env.define(vehicle)
    vehicle.store.define(Field("TBL", np.eye(3), "mat", "out", "kinematics"))
    vehicle.store.set("dvbe", dvbe)
    vehicle.store.set("alpha0x", alpha0x)
    vehicle.store.set("beta0x", beta0x)
    vehicle.store.set("sbel1", sbel[0])
    vehicle.store.set("sbel2", sbel[1])
    vehicle.store.set("sbel3", sbel[2])
    newton.initialize(vehicle, _ctx())
    return vehicle, env, newton


def test_name_is_environment():
    assert Sam6Environment().name == "environment"


def test_define_cpp_fields():
    vehicle = _Vehicle()
    Sam6Environment().define(vehicle)
    store = vehicle.store
    assert tuple(store.names()) == DEFINED
    for name in DEFINED:
        field = store.field(name)
        assert field.module == "environment"
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        if name in INT_FIELDS:
            assert field.type == "int"
            assert store.get(name) == 0
        else:
            assert field.type == "real"
            assert _approx(store.get(name), 0.0), name


def test_define_does_not_register_newton_missile_or_wind_fields():
    vehicle = _Vehicle()
    Sam6Environment().define(vehicle)
    for name in NOT_DEFINED:
        assert name not in vehicle.store.names()


def test_newton_init_then_env_matches_atmosphere76():
    vehicle, env, _newton = _newton_init_then_env()
    store = vehicle.store
    assert _approx(store.get("dvbe"), 16.0)
    assert _approx(store.get("alt"), 1000.0)
    assert _approx(store.get("hbe"), 1000.0)
    vbel = store.get("VBEL")
    assert np.all(np.isfinite(vbel))
    env.execute(vehicle, _ctx())
    assert _approx(store.get("hbe"), store.get("alt"))
    rho, press, tempk = atmosphere76(1000.0)
    vsound = (1.4 * R * tempk) ** 0.5
    np.testing.assert_allclose(store.get("rho"), rho, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("press"), press, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("tempk"), tempk, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("vsound"), vsound, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("grav"), _grav(1000.0), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        store.get("vmach"), abs(16.0 / vsound), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        store.get("pdynmc"), 0.5 * rho * 16.0 * 16.0, rtol=RTOL, atol=ATOL
    )


def test_hbe_equals_alt_after_newton_init():
    vehicle, env, _newton = _newton_init_then_env()
    assert _approx(vehicle.store.get("hbe"), vehicle.store.get("alt"))
    env.execute(vehicle, _ctx())
    assert _approx(vehicle.store.get("hbe"), 1000.0)
    assert _approx(vehicle.store.get("alt"), 1000.0)


def test_guid_term_6_tiny_pdynmc_sets_trcond_3():
    vehicle, env, _newton = _newton_init_then_env()
    store = vehicle.store
    store.define(Field("mguide", 6, "int", "data", "guidance"))
    store.define(Field("trdynm", 1e4, "real", "data", "aerodynamics"))
    store.define(Field("trcond", 0, "int", "diag", "aerodynamics"))
    env.execute(vehicle, _ctx())
    assert store.get("pdynmc") <= store.get("trdynm")
    assert store.get("trcond") == 3


def test_guid_term_16_is_decoded_as_6():
    vehicle, env, _newton = _newton_init_then_env()
    store = vehicle.store
    store.define(Field("mguide", 16, "int", "data", "guidance"))
    store.define(Field("trdynm", 1e4, "real", "data", "aerodynamics"))
    store.define(Field("trcond", 0, "int", "diag", "aerodynamics"))
    env.execute(vehicle, _ctx())
    assert store.get("trcond") == 3


def test_guid_term_not_6_leaves_trcond():
    vehicle, env, _newton = _newton_init_then_env()
    store = vehicle.store
    store.define(Field("mguide", 0, "int", "data", "guidance"))
    store.define(Field("trdynm", 1e4, "real", "data", "aerodynamics"))
    store.define(Field("trcond", 0, "int", "diag", "aerodynamics"))
    env.execute(vehicle, _ctx())
    assert store.get("trcond") == 0


def test_guid_term_6_pdynmc_above_trdynm_leaves_trcond():
    vehicle, env, _newton = _newton_init_then_env()
    store = vehicle.store
    store.define(Field("mguide", 6, "int", "data", "guidance"))
    store.define(Field("trdynm", 1e-12, "real", "data", "aerodynamics"))
    store.define(Field("trcond", 0, "int", "diag", "aerodynamics"))
    env.execute(vehicle, _ctx())
    assert store.get("pdynmc") > store.get("trdynm")
    assert store.get("trcond") == 0


def test_skips_trcond_when_mguide_absent():
    vehicle, env, _newton = _newton_init_then_env()
    env.execute(vehicle, _ctx())
    assert "mguide" not in vehicle.store.names()
    assert "trcond" not in vehicle.store.names()
    assert np.isfinite(vehicle.store.get("vmach"))


def test_skips_mfreeze_when_absent():
    vehicle, env, _newton = _newton_init_then_env()
    assert "mfreeze" not in vehicle.store.names()
    env.execute(vehicle, _ctx())
    live = vehicle.store.get("vmach")
    vehicle.store.set("dvbe", 32.0)
    env.execute(vehicle, _ctx())
    assert vehicle.store.get("vmach") != pytest.approx(live, rel=RTOL, abs=ATOL)


def test_mfreeze_latches_vmach_and_pdynmc_when_present():
    vehicle, env, _newton = _newton_init_then_env()
    store = vehicle.store
    store.define(Field("mfreeze", 1, "int", "data", "control"))
    env.execute(vehicle, _ctx())
    frozen_mach = store.get("vmach")
    frozen_q = store.get("pdynmc")
    store.set("dvbe", 32.0)
    env.execute(vehicle, _ctx())
    assert _approx(store.get("vmach"), frozen_mach)
    assert _approx(store.get("pdynmc"), frozen_q)
    assert store.get("mfreeze_environ") == 1
    assert _approx(store.get("machf"), frozen_mach)
    assert _approx(store.get("pdynmcf"), frozen_q)


def test_importing_environment_does_not_import_flat6_classes():
    import cadac.vehicles.sam6.environment as envmod

    assert "Flat6Environment" not in dir(envmod)
    from cadac.eom.flat6 import Flat6Environment

    assert not issubclass(Sam6Environment, Flat6Environment)


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.sam6.environment as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6Environment" not in src
    assert "from cadac.eom.flat6 import" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src
    assert "atmosphere76" in src
    assert "EARTH_MASS" in src
    assert "REARTH" in src


def test_initialize_and_terminate_are_pass():
    vehicle, env, _newton = _newton_init_then_env()
    assert env.initialize(vehicle, _ctx()) is None
    assert env.terminate(vehicle, _ctx()) is None

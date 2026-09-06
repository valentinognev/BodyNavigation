import inspect

import numpy as np
from cadac.env.us76 import atmosphere76
from cadac.vehicles.sam6.flat3 import Sam6Flat3Environment, Sam6Flat3Newton
from cadac.kernel.state import StateStore, Field
from cadac.kernel.executive import SimContext

class V:
    def __init__(self):
        self.store = StateStore()

def test_env_mach_from_dvae():
    v = V()
    e = Sam6Flat3Environment()
    e.define(v)
    v.store.define(Field("dvae", 250.0, "real", "data", "newton"))
    v.store.define(Field("SAEL", (0.0, 0.0, -10000.0), "vec", "state", "newton"))
    e.execute(v, SimContext(0.0, 0.01, 0.0, 0.0, [], 0))
    rho, press, tempk = atmosphere76(10000.0)
    np.testing.assert_allclose(v.store.get("rho"), rho, rtol=1e-12, atol=1e-14)
    assert v.store.get("mach") > 0.0

def test_newton_init_sael():
    v = V()
    n = Sam6Flat3Newton()
    n.define(v)
    v.store.set("dvae", 250.0)
    v.store.set("psivlx", 90.0)
    v.store.set("thtvlx", 0.0)
    v.store.set("sael1", 0.0)
    v.store.set("sael2", -30000.0)
    v.store.set("sael3", -10000.0)
    n.initialize(v, SimContext(0.0, 0.01, 0.0, 0.0, [], 0))
    np.testing.assert_allclose(v.store.get("SAEL"), [0.0, -30000.0, -10000.0], rtol=1e-12)
    np.testing.assert_allclose(v.store.get("alt"), 10000.0, rtol=1e-12, atol=1e-14)


from cadac.constants import DEG, R, RAD
from cadac.env.gravity import gravity
from cadac.kernel.integrate import integrate
from cadac.math.frames import mat2tr, polar_from_cart
from cadac.vehicles.sam6.flat3 import Sam6Flat3Kinematics


def test_module_names():
    assert Sam6Flat3Environment().name == "environment"
    assert Sam6Flat3Kinematics().name == "kinematics"
    assert Sam6Flat3Newton().name == "newton"


def test_kinematics_defines_cpp_fields():
    v = V()
    k = Sam6Flat3Kinematics()
    k.define(v)
    time = v.store.field("time")
    assert time.type == "real"
    assert time.role == "exec"
    assert time.module == "kinematics"
    assert time.outputs == ("com",)
    delay = v.store.field("launch_delay")
    assert delay.type == "real"
    assert delay.role == "data"
    epoch = v.store.field("launch_epoch")
    assert epoch.type == "real"
    assert epoch.role == "out"
    assert epoch.outputs == ("com",)
    elapsed = v.store.field("launch_time")
    assert elapsed.type == "real"
    assert elapsed.role == "diag"


def test_kinematics_launch_clock():
    v = V()
    k = Sam6Flat3Kinematics()
    k.define(v)
    v.store.set("launch_delay", 12.0)
    ctx = SimContext(0.0, 0.01, 0.0, 0.0, [], 0)
    k.initialize(v, ctx)
    assert v.store.get("time") == 0.0
    np.testing.assert_allclose(v.store.get("launch_epoch"), 12.0, rtol=1e-12, atol=1e-14)
    ctx = SimContext(15.0, 0.01, 0.0, 0.0, [], 0)
    k.execute(v, ctx)
    assert v.store.get("time") == 15.0
    np.testing.assert_allclose(v.store.get("launch_time"), 3.0, rtol=1e-12, atol=1e-14)


def test_environment_does_not_define_newton_names_or_vmach():
    v = V()
    Sam6Flat3Environment().define(v)
    for name in ("alt", "SAEL", "dvae", "vmach"):
        assert name not in v.store.names()
    for name in ("grav", "rho", "pdynmc", "mach", "vsound", "press"):
        assert v.store.get(name) == 0.0


def test_env_formulas_from_dvae_and_sael():
    v = V()
    e = Sam6Flat3Environment()
    e.define(v)
    v.store.define(Field("dvae", 250.0, "real", "init/out", "newton"))
    v.store.define(Field("SAEL", (0.0, 0.0, -10000.0), "vec", "state", "newton"))
    e.execute(v, SimContext(0.0, 0.01, 0.0, 0.0, [], 0))
    rho, press, tempk = atmosphere76(10000.0)
    vsound = (1.4 * R * tempk) ** 0.5
    np.testing.assert_allclose(v.store.get("grav"), gravity(10000.0), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(v.store.get("rho"), rho, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(v.store.get("press"), press, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(v.store.get("vsound"), vsound, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(v.store.get("mach"), abs(250.0 / vsound), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(v.store.get("pdynmc"), 0.5 * rho * 250.0**2, rtol=1e-12, atol=1e-14)
    assert "vmach" not in v.store.names()


def test_newton_does_not_define_environment_or_forces():
    v = V()
    Sam6Flat3Newton().define(v)
    assert "alt" in v.store.names()
    for name in ("grav", "rho", "pdynmc", "mach", "vsound", "press", "FSPA", "phiavout"):
        assert name not in v.store.names()


def test_newton_init_vael_east_without_phiavout():
    v = V()
    n = Sam6Flat3Newton()
    n.define(v)
    v.store.set("dvae", 250.0)
    v.store.set("psivlx", 90.0)
    v.store.set("thtvlx", 0.0)
    v.store.set("sael1", 0.0)
    v.store.set("sael2", -30000.0)
    v.store.set("sael3", -10000.0)
    n.initialize(v, SimContext(0.0, 0.01, 0.0, 0.0, [], 0))
    psivl = 90.0 * RAD
    expected = np.array(
        [
            250.0 * np.cos(0.0) * np.cos(psivl),
            250.0 * np.cos(0.0) * np.sin(psivl),
            250.0 * (-np.sin(0.0)),
        ]
    )
    np.testing.assert_allclose(v.store.get("VAEL"), expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(v.store.get("TAV"), np.eye(3), rtol=1e-12, atol=1e-14)
    tvl = mat2tr(psivl, 0.0)
    np.testing.assert_allclose(v.store.get("TVL"), tvl, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(v.store.get("TAL"), tvl, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(v.store.get("psivl"), psivl, rtol=1e-12, atol=1e-14)


def test_newton_execute_next_acc_from_tal_fspa():
    v = V()
    n = Sam6Flat3Newton()
    n.define(v)
    v.store.set("dvae", 250.0)
    v.store.set("psivlx", 90.0)
    v.store.set("thtvlx", 0.0)
    v.store.set("sael1", 0.0)
    v.store.set("sael2", -30000.0)
    v.store.set("sael3", -10000.0)
    v.store.define(Field("FSPA", (0.0, 0.0, 0.0), "vec", "out", "forces"))
    v.store.define(Field("grav", gravity(10000.0), "real", "out", "environment"))
    n.initialize(v, SimContext(0.0, 0.01, 0.0, 0.0, [], 0))
    tal = v.store.get("TAL").copy()
    fspa = v.store.get("FSPA").copy()
    grav = v.store.get("grav")
    sael_old = v.store.get("SAEL").copy()
    vael_old = v.store.get("VAEL").copy()
    aael_old = v.store.get("AAEL").copy()
    dt = 0.01
    n.execute(v, SimContext(0.0, dt, 0.0, 0.0, [], 0))
    assert v.store.get("SAEL")[2] > sael_old[2]
    assert v.store.get("SAEL")[2] < 0.0
    next_acc = tal.T @ fspa + np.array([0.0, 0.0, grav])
    np.testing.assert_allclose(v.store.get("AAEL"), next_acc, rtol=1e-12, atol=1e-14)
    next_vel = integrate(next_acc, aael_old, vael_old, dt)
    sael_expected = integrate(next_vel, vael_old, sael_old, dt)
    np.testing.assert_allclose(v.store.get("VAEL"), next_vel, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(v.store.get("SAEL"), sael_expected, rtol=1e-12, atol=1e-14)
    polar = polar_from_cart(next_vel)
    tvl = mat2tr(polar[1], polar[2])
    tav = np.eye(3)
    tal_expected = tav @ tvl
    np.testing.assert_allclose(v.store.get("TVL"), tvl, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(v.store.get("TAV"), tav, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(v.store.get("TAL"), tal_expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(v.store.get("dvae"), polar[0], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(v.store.get("psivl"), polar[1], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(v.store.get("thtvl"), polar[2], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(v.store.get("psivlx"), polar[1] * DEG, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(v.store.get("thtvlx"), polar[2] * DEG, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(v.store.get("alt"), -sael_expected[2], rtol=1e-12, atol=1e-14)


def test_does_not_import_flat6_or_shared_flat3():
    import cadac.vehicles.sam6.flat3 as sam6_flat3

    src = inspect.getsource(sam6_flat3)
    assert "cadac.eom.flat6" not in src
    assert "cadac.eom.flat3" not in src
    assert "Flat6Environment" not in dir(sam6_flat3)
    assert "Flat3Newton" not in dir(sam6_flat3)

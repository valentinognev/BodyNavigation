import numpy as np

from cadac.constants import DEG
from cadac.env.gravity import gravity
from cadac.eom.round3 import Round3Newton
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.earth import cadsph, cadtei, cadtge
from cadac.math.frames import mat2tr, polar_from_cart


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(**kwargs):
    fields = dict(
        sim_time=0.0,
        int_step=0.1,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )
    fields.update(kwargs)
    return SimContext(**fields)


def _defined_newton():
    vehicle = _Vehicle()
    newton = Round3Newton()
    newton.define(vehicle)
    return vehicle, newton


def _hyper3_ics(store):
    store.set("lonx", -80.55)
    store.set("latx", 28.43)
    store.set("alt", 3000.0)
    store.set("psivgx", 90.0)
    store.set("thtvgx", 0.0)
    store.set("dvbe", 250.0)


def _ready_step(dt=0.01):
    vehicle, newton = _defined_newton()
    _hyper3_ics(vehicle.store)
    vehicle.store.define(Field("FSPV", (0.0, 0.0, 0.0), "vec", "out", "forces"))
    vehicle.store.define(Field("grav", gravity(3000.0), "real", "out", "environment"))
    ctx = _ctx(sim_time=0.0, int_step=dt)
    newton.initialize(vehicle, ctx)
    return vehicle, newton, ctx


def test_execute_zero_fspv_one_step_integrates_vbii_then_sbii():
    vehicle, newton, ctx = _ready_step(dt=0.01)
    store = vehicle.store
    sbii_old = store.get("sbii").copy()
    vbii_old = store.get("vbii").copy()
    abii_old = store.get("abii").copy()
    vbeg_old = store.get("vbeg").copy()
    sbeg_old = store.get("sbeg").copy()
    weii = store.get("weii").copy()
    tig = store.get("tig").copy()
    tgv = store.get("tgv").copy()
    fspv = store.get("FSPV").copy()
    grav = store.get("grav")
    dt = ctx.int_step

    newton.execute(vehicle, ctx)

    assert np.isfinite(store.get("alt"))
    assert not np.allclose(store.get("sbii"), sbii_old, rtol=1e-12, atol=1e-14)

    grav_vec = np.array([0.0, 0.0, grav])
    abii_new = tig @ ((tgv @ fspv) + grav_vec)
    vbii_expected = integrate(abii_new, abii_old, vbii_old, dt)
    sbii_expected = integrate(vbii_expected, vbii_old, sbii_old, dt)
    np.testing.assert_allclose(store.get("vbii"), vbii_expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("sbii"), sbii_expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("abii"), abii_new, rtol=1e-12, atol=1e-14)

    tei = cadtei(ctx.sim_time)
    lon, lat, alt = cadsph(tei @ sbii_expected)
    tge = cadtge(lon, lat)
    tgi = tge @ tei
    vbeg_new = tgi @ (vbii_expected - weii @ sbii_expected)
    sbeg_expected = integrate(vbeg_new, vbeg_old, sbeg_old, dt)
    polar = polar_from_cart(vbeg_new)
    tig_expected = tgi.T
    tgv_expected = mat2tr(polar[1], polar[2]).T

    np.testing.assert_allclose(store.get("tge"), tge, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("vbeg"), vbeg_new, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("sbeg"), sbeg_expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("tig"), tig_expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("tgv"), tgv_expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("lonx"), lon * DEG, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("latx"), lat * DEG, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("alt"), alt, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("altx"), alt / 1000.0, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("dvbe"), polar[0], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("psivg"), polar[1], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("thtvg"), polar[2], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("psivgx"), polar[1] * DEG, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(store.get("thtvgx"), polar[2] * DEG, rtol=1e-12, atol=1e-14)

import math
from types import SimpleNamespace

import numpy as np

from cadac.constants import AGRAV, DEG, RAD, REARTH
from cadac.eom.round6 import Round6Newton
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat2tr, mat3tr, polar_from_cart
from cadac.math.wgs84 import cad_geo84_in, cad_grav84, cad_tdi84, cad_tgi84

_ZEROS3 = (0.0, 0.0, 0.0)
_ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
_RTOL = 1e-12
_ATOL = 1e-14
_VMASS = 136077.0
_FOOT = 3.280834
_NMILES = 5.399568e-4


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


def _climb_ics(store):
    store.set("minit", 0)
    store.set("lonx", 10.0)
    store.set("latx", 10.0)
    store.set("alt", 10000.0)
    store.set("dvbe", 1000.0)
    store.set("alpha0x", 2.5)
    store.set("beta0x", 0.0)
    _plant_kinematics(store)


def _plant_step_inputs(store, *, fapb=(0.0, 0.0, 0.0), vmass=_VMASS):
    tbd = mat3tr(
        store.get("psibdx") * RAD,
        store.get("thtbdx") * RAD,
        store.get("phibdx") * RAD,
    )
    tdi = store.get("TDI")
    tbi = tbd @ tdi
    gravg = cad_grav84(store.get("SBII"), store.get("time"))
    if "TBI" not in store.names():
        store.define(Field("TBI", _ZEROS33, "mat", "state", "kinematics"))
    if "GRAVG" not in store.names():
        store.define(Field("GRAVG", _ZEROS3, "vec", "out", "environment"))
    if "FAPB" not in store.names():
        store.define(Field("FAPB", _ZEROS3, "vec", "out", "forces"))
    if "vmass" not in store.names():
        store.define(Field("vmass", 0.0, "real", "init", "propulsion"))
    store.set("TBI", tbi)
    store.set("GRAVG", gravg)
    store.set("FAPB", np.asarray(fapb, dtype=float))
    store.set("vmass", float(vmass))


def _ready_step(dt=0.01, fapb=(0.0, 0.0, 0.0), vmass=_VMASS):
    vehicle, newton = _vehicle()
    _climb_ics(vehicle.store)
    ctx = _ctx(sim_time=0.0, int_step=dt)
    newton.initialize(vehicle, ctx)
    _plant_step_inputs(vehicle.store, fapb=fapb, vmass=vmass)
    return vehicle, newton, ctx


def _cpp_newton_step(store, dt):
    tdi = store.get("TDI").copy()
    tgi = store.get("TGI").copy()
    weii = store.get("WEII").copy()
    grndtrck = float(store.get("grndtrck"))
    sbii = store.get("SBII").copy()
    vbii = store.get("VBII").copy()
    abii = store.get("ABII").copy()
    time = store.get("time")
    gravg = store.get("GRAVG").copy()
    tbi = store.get("TBI").copy()
    fapb = store.get("FAPB").copy()
    vmass = store.get("vmass")

    fspb = fapb * (1.0 / vmass)
    next_acc = tbi.T @ fspb + tgi.T @ gravg
    next_vel = integrate(next_acc, abii, vbii, dt)
    sbii = integrate(next_vel, vbii, sbii, dt)
    abii = next_acc
    vbii = next_vel
    dvbi = float(np.linalg.norm(vbii))
    dbi = float(np.linalg.norm(sbii))

    lon, lat, alt = cad_geo84_in(sbii, time)
    tdi = cad_tdi84(lon, lat, alt, time)
    tgi = cad_tgi84(lon, lat, alt, time)
    lonx = lon * DEG
    latx = lat * DEG
    altx = 0.001 * alt * _FOOT

    vbed = tdi @ (vbii - weii @ sbii)
    polar = polar_from_cart(vbed)
    dvbe = float(polar[0])
    psivdx = DEG * float(polar[1])
    thtvdx = DEG * float(polar[2])
    tvd = mat2tr(psivdx * RAD, thtvdx * RAD)

    ayx = fspb[1] / AGRAV
    anx = -fspb[2] / AGRAV
    grndtrck += math.sqrt(vbed[0] * vbed[0] + vbed[1] * vbed[1]) * dt * REARTH / dbi
    gndtrkmx = 0.001 * grndtrck
    gndtrnmx = _NMILES * grndtrck

    return {
        "FSPB": fspb,
        "ABII": abii,
        "VBII": vbii,
        "SBII": sbii,
        "dvbi": dvbi,
        "dbi": dbi,
        "lonx": lonx,
        "latx": latx,
        "alt": alt,
        "altx": altx,
        "TDI": tdi,
        "TGI": tgi,
        "VBED": vbed,
        "dvbe": dvbe,
        "psivdx": psivdx,
        "thtvdx": thtvdx,
        "TVD": tvd,
        "ayx": ayx,
        "anx": anx,
        "grndtrck": grndtrck,
        "gndtrkmx": gndtrkmx,
        "gndtrnmx": gndtrnmx,
    }


def test_execute_zero_fapb_one_step_alt_finite_sbii_changes_abii_matches_next_acc():
    vehicle, newton, ctx = _ready_step(dt=0.01, fapb=(0.0, 0.0, 0.0))
    store = vehicle.store
    sbii_old = store.get("SBII").copy()
    want = _cpp_newton_step(store, ctx.int_step)

    newton.execute(vehicle, ctx)

    assert np.isfinite(store.get("alt"))
    assert not np.allclose(store.get("SBII"), sbii_old, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("ABII"), want["ABII"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("FSPB"), want["FSPB"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("VBII"), want["VBII"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("SBII"), want["SBII"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("VBED"), want["VBED"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("TDI"), want["TDI"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("TGI"), want["TGI"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("TVD"), want["TVD"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("lonx"), want["lonx"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("latx"), want["latx"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("alt"), want["alt"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("altx"), want["altx"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("dvbe"), want["dvbe"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("dvbi"), want["dvbi"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("dbi"), want["dbi"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("psivdx"), want["psivdx"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("thtvdx"), want["thtvdx"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("ayx"), want["ayx"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("anx"), want["anx"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(
        store.get("grndtrck"), want["grndtrck"], rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        store.get("gndtrkmx"), want["gndtrkmx"], rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        store.get("gndtrnmx"), want["gndtrnmx"], rtol=_RTOL, atol=_ATOL
    )
    assert "mfreeze" not in store.names()
    assert "ABEL" not in store.names()


def test_execute_nonzero_fapb_fspb_and_next_acc_match_cpp():
    fapb = (_VMASS, 0.0, 0.0)
    vehicle, newton, ctx = _ready_step(dt=0.01, fapb=fapb)
    store = vehicle.store
    want = _cpp_newton_step(store, ctx.int_step)

    newton.execute(vehicle, ctx)

    np.testing.assert_allclose(store.get("FSPB"), np.array([1.0, 0.0, 0.0]), rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("ABII"), want["ABII"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("VBII"), want["VBII"], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(store.get("SBII"), want["SBII"], rtol=_RTOL, atol=_ATOL)

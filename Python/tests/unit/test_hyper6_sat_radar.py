import numpy as np
import pytest

from cadac.cli import _resolve_vehicle
from cadac.constants import RAD, REARTH, WEII3
from cadac.eom.round6 import _cad_in_orb
from cadac.kernel.combus import packet_from_store
from cadac.kernel.executive import SimContext, run_loop
from cadac.math.earth import cadtei, cadtge
from cadac.math.frames import cadac_matmul, cart_from_pol, polar_from_cart

RTOL = 1e-12

_ORBIT_CARDS = {
    "minit": 1,
    "semi": 7000000.0,
    "ecc": 0.0622,
    "inclx": 48.78,
    "lon_anodex": 254.0,
    "arg_perix": 43.3,
    "true_anomx": -51.3,
}


def _run_one_step(veh):
    ctx = SimContext(
        sim_time=0.0,
        int_step=0.01,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )
    for module in veh.modules:
        module.initialize(veh, ctx)
    run_loop(
        [veh],
        {veh: veh.modules},
        [module.name for module in veh.modules],
        0.0,
        0.01,
    )


def test_sat3_radar_registered():
    assert _resolve_vehicle(None, "SAT3").__name__ == "Hyper6Satellite"
    assert _resolve_vehicle(None, "RADAR0").__name__ == "Hyper6Radar"


def test_sat3_one_step_advances_orbit():
    from cadac.tables.lookup import Datadeck

    assert Datadeck is not None
    cls = _resolve_vehicle(None, "SAT3")
    veh = cls("sat", events=[])
    veh.define()
    _run_one_step(veh)
    assert veh.store.get("time") > 0


def _ctx(sim_time=0.0):
    return SimContext(
        sim_time=sim_time,
        int_step=0.01,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def test_radar_seeker_track_from_sat3_packet():
    sat = _resolve_vehicle(None, "SAT3")("sat", events=[])
    sat.define()
    stii = np.array([7000000.0, 1000.0, -2000.0])
    vtii = np.array([0.0, 7500.0, 10.0])
    sat.store.set("sbii", stii)
    sat.store.set("vbii", vtii)
    packet = packet_from_store(sat.store, sat.com_names)
    packet.type = sat.type
    radar = _resolve_vehicle(None, "RADAR0")("rdr", events=[])
    radar.define()
    radar.store.set("radar_on", 1)
    radar.store.set("lonx", -25.0)
    radar.store.set("latx", 37.0)
    radar.store.set("dat_sigma", 0.0)
    radar.store.set("azat_sigma", 0.0)
    radar.store.set("elat_sigma", 0.0)
    radar.store.set("vel_sigma", 0.0)
    ctx = _ctx()
    radar.modules[0].initialize(radar, ctx)
    ctx.combus = [packet]
    radar.modules[1].execute(radar, ctx)
    sbii = radar.store.get("sbii")
    polar = polar_from_cart(sbii - stii)
    expected_stc = sbii - cart_from_pol(float(polar[0]), float(polar[1]), float(polar[2]))
    np.testing.assert_allclose(radar.store.get("stcii1"), expected_stc, rtol=RTOL, atol=0.0)
    np.testing.assert_allclose(radar.store.get("vtcii1"), vtii, rtol=RTOL, atol=0.0)


def test_sat3_minit1_matches_cad_in_orb():
    veh = _resolve_vehicle(None, "SAT3")("Satellite", events=[])
    veh.define()
    for name, value in _ORBIT_CARDS.items():
        veh.store.set(name, value)
    newton = next(module for module in veh.modules if module.name == "newton")
    newton.initialize(veh, _ctx())
    sbii, vbii, _flag = _cad_in_orb(
        _ORBIT_CARDS["semi"],
        _ORBIT_CARDS["ecc"],
        _ORBIT_CARDS["inclx"],
        _ORBIT_CARDS["lon_anodex"],
        _ORBIT_CARDS["arg_perix"],
        _ORBIT_CARDS["true_anomx"],
    )
    np.testing.assert_allclose(veh.store.get("sbii"), sbii, rtol=RTOL, atol=0.0)
    np.testing.assert_allclose(veh.store.get("vbii"), vbii, rtol=RTOL, atol=0.0)


def test_radar_ground_kinematics_matches_cadac_matmul():
    lonx, latx, alt, sim_time = -25.0, 37.0, 100.0, 10.0
    radar = _resolve_vehicle(None, "RADAR0")("rdr", events=[])
    radar.define()
    radar.store.set("lonx", lonx)
    radar.store.set("latx", latx)
    radar.store.set("alt", alt)
    radar.modules[0].initialize(radar, _ctx(sim_time))
    weii = np.zeros((3, 3))
    weii[0, 1] = -WEII3
    weii[1, 0] = WEII3
    sbig = np.array([0.0, 0.0, -(alt + REARTH)])
    tge = cadtge(lonx * RAD, latx * RAD)
    sbie = cadac_matmul(tge.T, sbig)
    tei = cadtei(sim_time)
    sbii = cadac_matmul(tei.T, sbie)
    vbii = cadac_matmul(weii, sbii)
    tig = cadac_matmul(tei.T, tge.T)
    np.testing.assert_allclose(radar.store.get("sbii"), sbii, rtol=RTOL, atol=0.0)
    np.testing.assert_allclose(radar.store.get("vbii"), vbii, rtol=RTOL, atol=0.0)
    np.testing.assert_allclose(radar.store.get("tig"), tig, rtol=RTOL, atol=0.0)


def test_radar_holds_ground_tracks_not_a_vehicle():
    cls = _resolve_vehicle(None, "RADAR0")
    veh = cls("rdr", events=[])
    veh.define()
    assert "NGROUND0" in veh.store or "ground" in " ".join(veh.store.names()).lower()
    with pytest.raises(ValueError, match="Ground0"):
        _resolve_vehicle(None, "GROUND0")

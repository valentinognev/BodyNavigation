from math import asin, cos, exp, fabs, sin, sqrt, tan
from types import SimpleNamespace

import numpy as np

from cadac.constants import DEG, RAD
from cadac.math.frames import angle, cadac_matmul, mat2tr, polar_from_cart, skew
from cadac.math.wgs84 import cad_in_geo84
from cadac.tables.lookup import Datadeck
from cadac.vehicles.round6.hyper6.vehicle import Hyper6

RTOL = 1e-12
WP = (0.1, 0.05, 15000.0)


def _sign(variable):
    if variable < 0.0:
        return -1
    return 1


def _execute_module(veh, name, method="execute", ctx=None):
    if ctx is None:
        ctx = SimpleNamespace(int_step=0.01)
    for module in veh.modules:
        if module.name == name:
            getattr(module, method)(veh, ctx)
            return
    raise AssertionError(f"module {name!r} not on vehicle")


def _close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=0.0)


def _hyper6_vehicle(mguide, wp=WP):
    deck = Datadeck.from_tables([])
    veh = Hyper6("hyper6", aero_deck=deck, prop_deck=deck, events=[])
    veh.define()
    store = veh.store
    lon, lat, alt = wp
    store.set("mguide", mguide)
    store.set("wp_lonx", lon)
    store.set("wp_latx", lat)
    store.set("wp_alt", alt)
    store.set("time", 0.0)
    store.set("grav", 9.81)
    store.set("line_gain", 0.05)
    store.set("nl_gain_fact", 0.0)
    store.set("decrement", 1.0e5)
    store.set("psifdx", 0.0)
    store.set("thtfdx", 0.0)
    store.set("philimx", 45.0)
    store.set("thtvdcx", 0.0)
    store.set("alcomx", 7.0)
    store.set("ancomx", 7.0)
    store.set("phicomx", 7.0)
    return veh


def _arm_waypoint(veh, swbd, vbecd, dvbec, tdci=None, **fields):
    """Place the vehicle so TDCI @ (SWII - SBIIC) equals ``swbd``."""
    store = veh.store
    time = fields.pop("time", store.get("time"))
    tdci = np.eye(3) if tdci is None else np.asarray(tdci, dtype=float)
    swii = cad_in_geo84(
        store.get("wp_lonx") * RAD,
        store.get("wp_latx") * RAD,
        store.get("wp_alt"),
        time,
    )
    disp = np.linalg.solve(tdci, np.asarray(swbd, dtype=float))
    store.set("time", time)
    store.set("TDCI", tdci)
    store.set("SBIIC", swii - disp)
    store.set("VBECD", np.asarray(vbecd, dtype=float))
    store.set("dvbec", float(dvbec))
    for name, value in fields.items():
        store.set(name, value)
    return veh


def _cpp_line(store):
    """Independent transcription of Hyper::guidance_line (guidance.cpp)."""
    line_gain = store.get("line_gain")
    nl_gain_fact = store.get("nl_gain_fact")
    decrement = store.get("decrement")
    time = store.get("time")
    grav = store.get("grav")
    sbiic = np.asarray(store.get("SBIIC"), dtype=float)
    vbecd = np.asarray(store.get("VBECD"), dtype=float)
    dvbec = store.get("dvbec")
    tdci = np.asarray(store.get("TDCI"), dtype=float)
    thtvdcx = store.get("thtvdcx")
    philimx = store.get("philimx")
    tfd = mat2tr(store.get("psifdx") * RAD, store.get("thtfdx") * RAD)
    swii = cad_in_geo84(
        store.get("wp_lonx") * RAD,
        store.get("wp_latx") * RAD,
        store.get("wp_alt"),
        time,
    )
    swbd = cadac_matmul(tdci, swii - sbiic)
    polar = polar_from_cart(swbd)
    wp_sltrange = float(polar[0])
    tod = mat2tr(float(polar[1]), float(polar[2]))
    swbg1 = float(swbd[0])
    swbg2 = float(swbd[1])
    wp_grdrange = sqrt(swbg1 * swbg1 + swbg2 * swbg2)
    vbeo = cadac_matmul(tod, vbecd)
    vbef = cadac_matmul(tfd, vbecd)
    nl_gain = nl_gain_fact * (1.0 - exp(-wp_sltrange / decrement))
    algv = np.array(
        [
            grav * sin(thtvdcx * RAD),
            line_gain * (-float(vbeo[1]) + nl_gain * float(vbef[1])),
            line_gain * (-float(vbeo[2]) + nl_gain * float(vbef[2]))
            - grav * cos(thtvdcx * RAD),
        ]
    )
    rad_min = dvbec * dvbec / (grav * tan(philimx * RAD))
    if wp_grdrange < 2.0 * rad_min:
        sh = np.array([swbg1, swbg2, 0.0])
        vh = np.array([float(vbecd[0]), float(vbecd[1]), 0.0])
        wp_flag = _sign(float(np.dot(vh, sh)))
    else:
        wp_flag = 0
    return {
        "algv": algv,
        "wp_sltrange": wp_sltrange,
        "nl_gain": nl_gain,
        "VBEO": vbeo,
        "VBEF": vbef,
        "wp_grdrange": wp_grdrange,
        "SWBD": swbd,
        "rad_min": rad_min,
        "wp_flag": wp_flag,
    }


def _cpp_arc(store):
    """Independent transcription of Hyper::guidance_arc (guidance.cpp)."""
    time = store.get("time")
    grav = store.get("grav")
    sbiic = np.asarray(store.get("SBIIC"), dtype=float)
    vbecd = np.asarray(store.get("VBECD"), dtype=float)
    dvbec = store.get("dvbec")
    tdci = np.asarray(store.get("TDCI"), dtype=float)
    philimx = store.get("philimx")
    swii = cad_in_geo84(
        store.get("wp_lonx") * RAD,
        store.get("wp_latx") * RAD,
        store.get("wp_alt"),
        time,
    )
    swbd = cadac_matmul(tdci, swii - sbiic)
    sh = np.array([float(swbd[0]), float(swbd[1]), 0.0])
    dwbh = sqrt(float(swbd[0]) * float(swbd[0]) + float(swbd[1]) * float(swbd[1]))
    vh = np.array([float(vbecd[0]), float(vbecd[1]), 0.0])
    uv = cadac_matmul(skew(vh), sh)
    psiwvx = DEG * float(angle(vh, sh))
    psiwvx = psiwvx * _sign(float(np.dot(uv, np.array([0.0, 0.0, 1.0]))))
    argument = 0.0
    if fabs(psiwvx) < 90.0:
        num = -2.0 * dvbec * dvbec * sin(psiwvx * RAD)
        denom = -grav * dwbh
        if denom != 0.0:
            argument = num / denom
        if fabs(argument) <= 1.0 and fabs(asin(argument)) < philimx * RAD:
            phicomx = DEG * asin(argument)
        else:
            phicomx = philimx * _sign(argument)
    else:
        phicomx = philimx * _sign(psiwvx)
    rad_geometric = 0.0
    if psiwvx != 0.0:
        rad_geometric = fabs(dwbh / (2.0 * sin(psiwvx * RAD)))
    rad_min = dvbec * dvbec / (grav * tan(philimx * RAD))
    if dwbh < 2.0 * rad_min:
        wp_flag = _sign(float(np.dot(vh, sh)))
    else:
        wp_flag = 0
    return {
        "phicomx": phicomx,
        "psiwvx": psiwvx,
        "argument": argument,
        "wp_grdrange": dwbh,
        "SWBD": swbd,
        "rad_min": rad_min,
        "rad_geometric": rad_geometric,
        "wp_flag": wp_flag,
    }


def _assert_line_diags(store, expected):
    _close(store.get("wp_sltrange"), expected["wp_sltrange"])
    _close(store.get("nl_gain"), expected["nl_gain"])
    _close(store.get("VBEO"), expected["VBEO"])
    _close(store.get("VBEF"), expected["VBEF"])
    _close(store.get("wp_grdrange"), expected["wp_grdrange"])
    _close(store.get("SWBD"), expected["SWBD"])
    _close(store.get("rad_min"), expected["rad_min"])
    assert store.get("wp_flag") == expected["wp_flag"]


def _assert_arc_diags(store, expected):
    _close(store.get("phicomx"), expected["phicomx"])
    _close(store.get("wp_grdrange"), expected["wp_grdrange"])
    _close(store.get("SWBD"), expected["SWBD"])
    _close(store.get("rad_min"), expected["rad_min"])
    _close(store.get("rad_geometric"), expected["rad_geometric"])
    assert store.get("wp_flag") == expected["wp_flag"]
    _close(store.get("alcomx"), 0.0)
    _close(store.get("ancomx"), 0.0)


def _line_vehicle(mguide):
    veh = _hyper6_vehicle(mguide)
    return _arm_waypoint(
        veh,
        swbd=np.array([800.0, 300.0, -50.0]),
        vbecd=np.array([180.0, -40.0, 5.0]),
        dvbec=250.0,
        tdci=mat2tr(0.35, -0.15),
        time=12.5,
        line_gain=0.05,
        nl_gain_fact=0.8,
        decrement=20000.0,
        psifdx=12.0,
        thtfdx=-4.0,
        philimx=30.0,
        thtvdcx=5.0,
        grav=9.81,
    )


def test_hyper6_line_guidance_matches_cpp():
    veh30 = _line_vehicle(30)
    expected = _cpp_line(veh30.store)
    assert expected["wp_flag"] == 1
    assert expected["nl_gain"] != 0.0
    _execute_module(veh30, "guidance")
    grav = veh30.store.get("grav")
    _close(veh30.store.get("alcomx"), expected["algv"][1] / grav)
    _close(veh30.store.get("ancomx"), 0.0)
    _close(veh30.store.get("phicomx"), 0.0)
    _assert_line_diags(veh30.store, expected)

    veh3 = _line_vehicle(3)
    expected3 = _cpp_line(veh3.store)
    _execute_module(veh3, "guidance")
    _close(veh3.store.get("alcomx"), 0.0)
    _close(veh3.store.get("ancomx"), -expected3["algv"][2] / veh3.store.get("grav"))
    _close(veh3.store.get("phicomx"), 0.0)
    _assert_line_diags(veh3.store, expected3)

    veh33 = _line_vehicle(33)
    expected33 = _cpp_line(veh33.store)
    _execute_module(veh33, "guidance")
    grav33 = veh33.store.get("grav")
    _close(veh33.store.get("alcomx"), expected33["algv"][1] / grav33)
    _close(veh33.store.get("ancomx"), -expected33["algv"][2] / grav33)
    _close(veh33.store.get("phicomx"), 0.0)
    _assert_line_diags(veh33.store, expected33)


def test_hyper6_arc_shallow_matches_cpp():
    vbecd = np.array([500.0, 10.0, 1.0])
    veh = _arm_waypoint(
        _hyper6_vehicle(4),
        swbd=np.array([80000.0, 1500.0, 20.0]),
        vbecd=vbecd,
        dvbec=float(np.linalg.norm(vbecd)),
        tdci=mat2tr(-0.2, 0.05),
        time=4.0,
        philimx=45.0,
        grav=9.81,
    )
    expected = _cpp_arc(veh.store)
    assert fabs(expected["psiwvx"]) < 90.0
    assert fabs(expected["argument"]) <= 1.0
    assert expected["wp_flag"] == 0
    _execute_module(veh, "guidance")
    _assert_arc_diags(veh.store, expected)


def test_hyper6_arc_argument_limit_and_wp_flag():
    vbecd = np.array([220.0, 180.0, 0.0])
    veh = _arm_waypoint(
        _hyper6_vehicle(4),
        swbd=np.array([1500.0, 600.0, 0.0]),
        vbecd=vbecd,
        dvbec=float(np.linalg.norm(vbecd)),
        tdci=mat2tr(0.1, 0.25),
        time=8.0,
        philimx=45.0,
        grav=9.81,
    )
    expected = _cpp_arc(veh.store)
    assert fabs(expected["psiwvx"]) < 90.0
    assert fabs(expected["argument"]) > 1.0
    assert expected["wp_flag"] == 1
    _execute_module(veh, "guidance")
    _assert_arc_diags(veh.store, expected)


def test_hyper6_arc_beyond_ninety_matches_cpp():
    veh = _arm_waypoint(
        _hyper6_vehicle(4),
        swbd=np.array([-5000.0, 200.0, 0.0]),
        vbecd=np.array([200.0, 0.0, 0.0]),
        dvbec=200.0,
        philimx=45.0,
        grav=9.81,
    )
    expected = _cpp_arc(veh.store)
    assert fabs(expected["psiwvx"]) >= 90.0
    _execute_module(veh, "guidance")
    _assert_arc_diags(veh.store, expected)

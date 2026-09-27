from math import cos, sqrt
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import AGRAV, DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round6.hyper6.control import SMALL, Hyper6Control

RTOL = 1e-12
ATOL = 1e-14
DELIMX = 20.0
HUGE_ANCOMX = 1.0e6
GMAX = 1.0e12
GMINX = -1.0e12
INT_STEP = 0.01


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=INT_STEP,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _hyper6_vehicle(**overrides):
    vehicle = SimpleNamespace(store=StateStore(), modules=[])
    ctrl = Hyper6Control()
    vehicle.modules.append(ctrl)
    ctrl.define(vehicle)
    store = vehicle.store
    zeros3 = (0.0, 0.0, 0.0)
    for field in (
        Field("phibdcx", 0.0, "real", "out", "ins"),
        Field("ppcx", 3.0, "real", "out", "ins"),
        Field("dllp", -2.5, "real", "out", "aerodynamics"),
        Field("dllda", 18.0, "real", "out", "aerodynamics"),
        Field("dyb", -6.0, "real", "out", "aerodynamics"),
        Field("dydr", 4.0, "real", "out", "aerodynamics"),
        Field("dnb", 1.5, "real", "out", "aerodynamics"),
        Field("dnr", -0.8, "real", "out", "aerodynamics"),
        Field("dndr", -3.0, "real", "out", "aerodynamics"),
        Field("rrcx", 2.0, "real", "out", "ins"),
        Field("qqcx", 0.0, "real", "out", "ins"),
        Field("dvbec", 1000.0, "real", "out", "ins"),
        Field("dvbe", 1000.0, "real", "out", "newton"),
        Field("dla", 8.0, "real", "out", "aerodynamics"),
        Field("dma", -2.0, "real", "out", "aerodynamics"),
        Field("dmq", -1.5, "real", "out", "aerodynamics"),
        Field("dmde", 8.0, "real", "out", "aerodynamics"),
        Field("FSPCB", (0.0, 0.0, 1.0), "vec", "out", "ins"),
        Field("grav", AGRAV, "real", "out", "environment"),
        Field("gmax", GMAX, "real", "out", "aerodynamics"),
        Field("gminx", GMINX, "real", "out", "aerodynamics"),
        Field("psivdcx", 0.0, "real", "out", "ins"),
        Field("altc", 0.0, "real", "out", "ins"),
        Field("VBECD", zeros3, "vec", "out", "ins"),
    ):
        store.define(field)
    store.set("mroll", 0)
    store.set("wrcl", 8.0)
    store.set("zrcl", 0.9)
    store.set("zetlagr", 1.1)
    store.set("dalimx", 20.0)
    store.set("delimx", DELIMX)
    store.set("drlimx", 20.0)
    store.set("philimx", 60.0)
    store.set("gainl", 1.0)
    store.set("gainp", 0.0)
    store.set("waclp", 2.0)
    store.set("zaclp", 0.7)
    store.set("paclp", 4.0)
    store.set("facthead", 0.0)
    store.set("gainalt", 0.01)
    store.set("gainaltrate", 0.5)
    store.set("rcomx", 0.0)
    store.set("alcomx", 0.0)
    store.set("ancomx", 0.0)
    store.set("psivdcomx", 0.0)
    store.set("altcom", 0.0)
    for name, value in overrides.items():
        store.set(name, value)
    return vehicle


def _execute_module(veh, name):
    module = next(item for item in veh.modules if item.name == name)
    module.execute(veh, _ctx())


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _clamp(value, limit):
    if abs(value) > limit:
        return limit * _sign(value)
    return value


def _lateral_phicomx(store):
    # control.cpp: phicomx=-DEG*gainl*alcomx*sign(fspb3)
    fspb3 = store.get("FSPCB")[2]
    phicomx = -DEG * store.get("gainl") * store.get("alcomx") * _sign(fspb3)
    return _clamp(phicomx, store.get("philimx"))


def _yaw_delrcx(store):
    zetlagr = store.get("zetlagr")
    dyb = store.get("dyb")
    dydr = store.get("dydr")
    dnb = store.get("dnb")
    dnr = store.get("dnr")
    dndr = store.get("dndr")
    dvbec = store.get("dvbec")
    zrate = -dyb / dvbec + dnb * dydr / (dvbec * dndr)
    aa = -dyb / dvbec - dnr
    bb = dnb + dyb * dnr / dvbec
    dum1 = aa - 2.0 * zetlagr * zetlagr * zrate
    dum2 = aa * aa - 4.0 * zetlagr * zetlagr * bb
    radix = dum1 * dum1 - dum2
    if radix < 0.0:
        radix = 0.0
    if abs(dndr) < SMALL:
        dndr = SMALL * _sign(dndr)
    grate = (-dum1 + sqrt(radix)) / (-dndr)
    delrcx = grate * (store.get("rrcx") - store.get("rcomx"))
    return _clamp(delrcx, store.get("drlimx"))


def _heading_bank(store):
    # control.cpp: gainpsi=(dvbec/grav)*zrcl*wrcl*(1.-zrcl*zrcl)*(1.+facthead)
    #              phicomx=gainpsi*(psivdcomx-psivdcx)
    gainpsi = (
        (store.get("dvbec") / store.get("grav"))
        * store.get("zrcl")
        * store.get("wrcl")
        * (1.0 - store.get("zrcl") * store.get("zrcl"))
        * (1.0 + store.get("facthead"))
    )
    phicomx = gainpsi * (store.get("psivdcomx") - store.get("psivdcx"))
    return gainpsi, _clamp(phicomx, store.get("philimx"))


def _normal_accel(store, ancomx, int_step):
    waclp = store.get("waclp")
    zaclp = store.get("zaclp")
    paclp = store.get("paclp")
    gainp = store.get("gainp")
    dla = store.get("dla")
    dma = store.get("dma")
    dmq = store.get("dmq")
    dmde = store.get("dmde")
    dvbec = store.get("dvbec")
    qqcx = store.get("qqcx")
    fspb3 = store.get("FSPCB")[2]
    zzd = store.get("zzd")
    zz = store.get("zz")
    gainfb3 = waclp * waclp * paclp / (dla * dmde)
    gainfb2 = (2.0 * zaclp * waclp + paclp + dmq - dla / dvbec) / dmde
    gainfb1 = (
        waclp * waclp
        + 2.0 * zaclp * waclp * paclp
        + dma
        + dmq * dla / dvbec
        - gainfb2 * dmde * dla / dvbec
    ) / (dla * dmde) - gainp
    zzd_new = AGRAV * ancomx + fspb3
    zz = zz + (zzd_new + zzd) * int_step / 2.0
    zzd = zzd_new
    dqc = -gainfb1 * (-fspb3) - gainfb2 * qqcx * RAD + gainfb3 * zz + gainp * zzd
    return dqc * DEG, np.array([gainfb1, gainfb2, gainfb3]), zz, zzd


def _altitude_ancomx(store):
    altrate = -store.get("VBECD")[2]
    eh = store.get("gainalt") * (store.get("altcom") - store.get("altc"))
    phibdcx = store.get("phibdcx")
    if phibdcx == 0:
        phibdcx = SMALL
    ancomx = (
        (1.0 / cos(phibdcx * RAD))
        * (store.get("gainaltrate") * (eh - altrate) + store.get("grav"))
        / AGRAV
    )
    if ancomx > store.get("gmax"):
        ancomx = store.get("gmax")
    if ancomx < store.get("gminx"):
        ancomx = store.get("gminx")
    return ancomx, altrate


def test_hyper6_lateral_accel_control():
    veh = _hyper6_vehicle(maut=30, alcomx=0.5)
    store = veh.store
    want_phi = _lateral_phicomx(store)
    want_delrcx = _yaw_delrcx(store)
    assert want_phi != 0.0
    assert abs(want_phi) < store.get("philimx")
    _execute_module(veh, "control")
    assert _approx(store.get("phicomx"), want_phi)
    assert _approx(store.get("delrcx"), want_delrcx)


def test_hyper6_heading_control():
    veh = _hyper6_vehicle(maut=40, psivdcomx=0.2)
    store = veh.store
    want_gainpsi, want_phi = _heading_bank(store)
    assert want_phi != 0.0
    assert abs(want_phi) < store.get("philimx")
    _execute_module(veh, "control")
    assert _approx(store.get("gainpsi"), want_gainpsi)
    assert _approx(store.get("phicomx"), want_phi)


def test_hyper6_normal_accel_saturates_delecx_at_delimx():
    veh = _hyper6_vehicle(maut=3, ancomx=HUGE_ANCOMX)
    store = veh.store
    raw, gainfp, zz, zzd = _normal_accel(store, HUGE_ANCOMX, INT_STEP)
    assert abs(raw) > DELIMX
    _execute_module(veh, "control")
    assert _approx(store.get("ancomx"), HUGE_ANCOMX)
    assert _approx(store.get("delecx"), _clamp(raw, DELIMX))
    assert _approx(store.get("zz"), zz)
    assert _approx(store.get("zzd"), zzd)
    np.testing.assert_allclose(store.get("GAINFP"), gainfp, rtol=RTOL, atol=ATOL)


def test_hyper6_altitude_saturates_delecx_at_delimx():
    veh = _hyper6_vehicle(maut=5, ancomx=HUGE_ANCOMX, altcom=1.0e8)
    store = veh.store
    ancomx, altrate = _altitude_ancomx(store)
    raw, gainfp, zz, zzd = _normal_accel(store, ancomx, INT_STEP)
    assert abs(ancomx) > HUGE_ANCOMX / 100.0
    assert abs(raw) > DELIMX
    _execute_module(veh, "control")
    assert _approx(store.get("ancomx"), ancomx)
    assert _approx(store.get("altrate"), altrate)
    assert _approx(store.get("delecx"), _clamp(raw, DELIMX))
    assert _approx(store.get("zz"), zz)
    assert _approx(store.get("zzd"), zzd)
    np.testing.assert_allclose(store.get("GAINFP"), gainfp, rtol=RTOL, atol=ATOL)


def test_hyper6_unported_pitch_digit_raises():
    for maut in (32, 42):
        veh = _hyper6_vehicle(maut=maut, alcomx=0.5, psivdcomx=0.2, qcomx=4.0)
        with pytest.raises(ValueError, match="unknown maut"):
            _execute_module(veh, "control")
        assert veh.store.get("delacx") == 0.0
        assert veh.store.get("delecx") == 0.0
        assert veh.store.get("delrcx") == 0.0

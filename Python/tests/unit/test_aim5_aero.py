from math import acos, atan2, cos, sin
from pathlib import Path

import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.aim5.aero import SMALL, Aim5Aero

AIM5 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/AIM5_250114/AIM5"
AERO = AIM5 / "aim5_aero_deck.asc"
RTOL = 1e-12
ATOL = 1e-14
# input_hori Missile
ALPHAX = 0.0
BETAX = 0.0
AREA = 0.01767
ALPMAX = 35.0
MASS = 63.8
DVAE = 269.0
ALT = 10000.0  # -sael3


def _cadac_sign(variable):
    if variable < 0:
        return -1
    return 1


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.002,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _deck():
    _, tables = parse_asc_deck(AERO)
    return Datadeck.from_tables(tables)


def _expected(deck, alphax, betax, mach, mprop, area, alpmax, mass, grav, pdynmc):
    alpha = alphax * RAD
    beta = betax * RAD
    alpp = acos(cos(alpha) * cos(beta))
    dum1 = np.tan(beta)
    dum2 = sin(alpha)
    if abs(dum2) < SMALL:
        dum2 = SMALL * _cadac_sign(dum2)
    phip = atan2(dum1, dum2)
    alppx = alpp * DEG
    cd_name = "cd_aim_on_vs_alpha_mach" if mprop else "cd_aim_off_vs_alpha_mach"
    claim = deck.look_up("cl_aim_vs_alpha_mach", alppx, mach)
    cdaim = deck.look_up(cd_name, alppx, mach)
    caaim = cdaim * cos(alpha) - claim * sin(alpha)
    cnpaim = cdaim * sin(alpha) + claim * cos(alpha)
    cnaim = abs(cnpaim) * cos(phip)
    cyaim = -abs(cnpaim) * sin(phip)
    claim_max = deck.look_up("cl_aim_vs_alpha_mach", alpmax, mach)
    cdaim_max = deck.look_up(cd_name, alpmax, mach)
    cnp_max = cdaim_max * sin(alpmax * RAD) + claim_max * cos(alpmax * RAD)
    falphax = abs(alphax)
    fbetax = abs(betax)
    if falphax < 10:
        cnalp = (0.123 + 0.013 * falphax) * DEG
    else:
        cnalp = 0.06 * falphax**0.625 * DEG
    if fbetax < 10:
        cybet = -(0.123 + 0.013 * fbetax) * DEG
    else:
        cybet = -0.06 * fbetax**0.625 * DEG
    gmax = (cnp_max * pdynmc * area) / (mass * grav)
    return {
        "alppx": alppx,
        "phipx": phip * DEG,
        "claim": claim,
        "cdaim": cdaim,
        "caaim": caaim,
        "cnpaim": cnpaim,
        "cnaim": cnaim,
        "cyaim": cyaim,
        "cnalp": cnalp,
        "cybet": cybet,
        "gmax": gmax,
    }


def _ready(*, mprop=1, alphax=ALPHAX, betax=BETAX):
    rho, press, tempk = atmosphere76(ALT)
    vsound = (1.4 * 287.053 * tempk) ** 0.5
    mach = abs(DVAE / vsound)
    pdynmc = 0.5 * rho * DVAE**2
    grav = gravity(ALT)
    vehicle = _Vehicle()
    aero = Aim5Aero(_deck())
    aero.define(vehicle)
    store = vehicle.store
    for name, value, module in (
        ("mach", mach, "environment"),
        ("alphax", alphax, "control"),
        ("betax", betax, "control"),
        ("mprop", mprop, "propulsion"),
        ("mass", MASS, "propulsion"),
        ("grav", grav, "environment"),
        ("pdynmc", pdynmc, "environment"),
    ):
        store.define(Field(name, value, "int" if name == "mprop" else "real", "out", module))
    store.set("area", AREA)
    store.set("alpmax", ALPMAX)
    return vehicle, aero, _deck(), mach, grav, pdynmc


def test_name_is_aerodynamics():
    assert Aim5Aero(_deck()).name == "aerodynamics"


def test_small_is_1e_7_not_in_cadac_constants():
    import cadac.constants as cadac_constants

    assert SMALL == 1e-7
    assert not hasattr(cadac_constants, "SMALL")


def test_hori_mprop1_matches_cpp_formulas():
    vehicle, aero, deck, mach, grav, pdynmc = _ready(mprop=1)
    aero.execute(vehicle, _ctx())
    want = _expected(deck, ALPHAX, BETAX, mach, 1, AREA, ALPMAX, MASS, grav, pdynmc)
    for name, value in want.items():
        np.testing.assert_allclose(vehicle.store.get(name), value, rtol=RTOL, atol=ATOL)


def test_mprop0_uses_cd_off_table():
    vehicle, aero, deck, mach, grav, pdynmc = _ready(mprop=0)
    aero.execute(vehicle, _ctx())
    want = _expected(deck, ALPHAX, BETAX, mach, 0, AREA, ALPMAX, MASS, grav, pdynmc)
    np.testing.assert_allclose(vehicle.store.get("cdaim"), want["cdaim"], rtol=RTOL, atol=ATOL)


def test_cnalp_high_alpha_piecewise():
    vehicle, aero, deck, mach, grav, pdynmc = _ready(mprop=1, alphax=20.0, betax=12.0)
    aero.execute(vehicle, _ctx())
    want = _expected(deck, 20.0, 12.0, mach, 1, AREA, ALPMAX, MASS, grav, pdynmc)
    np.testing.assert_allclose(vehicle.store.get("cnalp"), want["cnalp"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("cybet"), want["cybet"], rtol=RTOL, atol=ATOL)


def test_define_does_not_register_mach_or_alphax():
    vehicle = _Vehicle()
    Aim5Aero(_deck()).define(vehicle)
    names = vehicle.store.names()
    assert "area" in names and "alpmax" in names and "gmax" in names
    assert "mach" not in names and "alphax" not in names

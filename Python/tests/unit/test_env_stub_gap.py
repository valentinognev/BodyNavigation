"""Task 7: matmo==1, maero==2, minit==1, SAM6 mins 2/3, mterm==-1.

ROCKET6 mguide==5 is already ported; the LTG case is a regression only.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.cli import _resolve_vehicle
from cadac.math.wgs84 import GM, cad_in_geo84
from cadac.stoch import seed
from cadac.tables.lookup import Datadeck
from cadac.vehicles.flat6.sam6.ins import _gauss3_rtl

_GLOBAL_FAMILY = {
    "HYPER6": None,
    "ROCKET6": "rocket6",
    "SAM6": "sam6",
}


def _vehicle(program, vtype, **kw):
    family = _GLOBAL_FAMILY.get(program, program.lower())
    cls = _resolve_vehicle(family, vtype)
    return cls("probe", **kw)


def _new(program, vtype):
    deck = Datadeck.from_tables({})
    veh = _vehicle(program, vtype, aero_deck=deck, prop_deck=deck, events=[])
    veh.define()
    return veh


def _execute_module(veh, name, method="execute", ctx=None):
    if ctx is None:
        ctx = SimpleNamespace(int_step=0.01)
    for module in veh.modules:
        if module.name == name:
            getattr(module, method)(veh, ctx)
            return
    raise AssertionError(f"module {name!r} not on vehicle")


def _hyper6_vehicle(**sets):
    veh = _new("HYPER6", "HYPER6")
    for name, value in sets.items():
        veh.store.set(name, value)
    return veh


def _rocket6_vehicle(**sets):
    veh = _new("ROCKET6", "HYPER6")
    for name, value in sets.items():
        veh.store.set(name, value)
    return veh


def _sam6_missile(**sets):
    veh = _new("SAM6", "MISSILE6")
    for name, value in sets.items():
        veh.store.set(name, value)
    return veh


def _nasa_altitude(veh, alt_m=90_000.0):
    veh.store.set("alt", alt_m)
    veh.store.set("SBII", cad_in_geo84(0.0, 0.0, alt_m, 0.0))


# us76_nasa2002 at 90 km (above the public-domain 84.852 km cutoff).
_NASA_90KM = {
    "rho": 3.4164039443612214e-06,
    "press": 0.18358727676748274,
    "tempk": 186.87,
}


def _assert_nasa_90km(store):
    for name, want in _NASA_90KM.items():
        np.testing.assert_allclose(store.get(name), want, rtol=1e-12, atol=0.0)


def test_round6_matmo1_atmosphere():
    veh = _hyper6_vehicle(mair=100)
    _nasa_altitude(veh)
    _execute_module(veh, "environment")
    _assert_nasa_90km(veh.store)


def test_rocket6_matmo1_atmosphere():
    veh = _rocket6_vehicle(mair=100)
    _nasa_altitude(veh)
    _execute_module(veh, "environment")
    _assert_nasa_90km(veh.store)


def test_hyper6_matmo1_out_of_range_exits_without_warning_flag():
    veh = _hyper6_vehicle(mair=100)
    veh.store.set("warning_flag", 0)
    alt = 1_100_000.0
    veh.store.set("alt", alt)
    veh.store.set("SBII", cad_in_geo84(0.0, 0.0, alt, 0.0))
    with pytest.raises(ValueError, match="us76_nasa2002"):
        _execute_module(veh, "environment")
    assert veh.store.get("warning_flag") == 0
    assert veh.store.get("rho") == 0.0


def test_rocket6_matmo1_out_of_range_warns_and_continues():
    veh = _rocket6_vehicle(mair=100)
    veh.store.set("warning_flag", 0)
    alt = 1_100_000.0
    veh.store.set("alt", alt)
    veh.store.set("SBII", cad_in_geo84(0.0, 0.0, alt, 0.0))
    _execute_module(veh, "environment")
    assert veh.store.get("warning_flag") == 1
    assert veh.store.get("rho") == 0.0
    assert veh.store.get("press") == 0.0
    assert veh.store.get("tempk") == 0.0
    assert veh.store.get("vsound") == 0.0
    _execute_module(veh, "environment")
    assert veh.store.get("warning_flag") == 1
    assert veh.store.get("rho") == 0.0


def test_hyper6_maero2_transfer_vehicle():
    veh = _hyper6_vehicle(maero=2)
    veh.store.set("refa_st", 7.0)
    veh.store.set("caa", 0.4)
    _execute_module(veh, "aerodynamics")
    assert veh.store.get("refa") == 7.0
    assert veh.store.get("cx") == pytest.approx(-0.4)


def test_hyper6_maero1_still_requires_tables():
    veh = _hyper6_vehicle(maero=1)
    with pytest.raises(KeyError, match="cd0_vs_alpha_mach"):
        _execute_module(veh, "aerodynamics")


_SEMI = 7_000_000.0


def test_hyper6_minit1_satellite_ics():
    """Circular equatorial overhead point: lon/lat 0, heading 90 deg, radius = semi."""
    veh = _hyper6_vehicle(minit=1)
    veh.store.set("lonx", 45.0)
    veh.store.set("latx", 10.0)
    veh.store.set("sat_semi", _SEMI)
    veh.store.set("sat_ecc", 0.0)
    veh.store.set("sat_inclx", 0.0)
    veh.store.set("sat_lon_anodex", 0.0)
    veh.store.set("sat_arg_perix", 0.0)
    veh.store.set("sat_true_anomx", 0.0)
    veh.store.set("ranglex_l_t", 0.0)
    veh.store.set("headon_flag", 0)
    veh.store.set("tgo_insertion", 0.0)
    veh.store.set("latx_bias", 0.0)
    veh.store.set("dvbi_bias", 0.0)
    veh.store.set("dbi_bias", 0.0)
    veh.store.set("thtvdx_bias", 0.0)
    _execute_module(veh, "newton", method="initialize")
    v_circ = math.sqrt(GM / _SEMI)
    np.testing.assert_allclose(veh.store.get("lonx"), 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(veh.store.get("latx"), 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(veh.store.get("psibdx"), 90.0, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(veh.store.get("dbi_desired"), _SEMI, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        veh.store.get("dvbi_desired"), v_circ, rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(veh.store.get("thtvdx_desired"), 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(veh.store.get("wp_lonx"), 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(veh.store.get("wp_latx"), 0.0, rtol=0.0, atol=0.0)


def test_rocket6_ltg_regression():
    veh = _rocket6_vehicle(mguide=5)
    _execute_module(veh, "guidance")
    assert veh.store.get("init_flag") == 0


# C++ Missile::init_ins BSpec / GSpec (mins 2 / 3). EUNBG is zero for BSpec.
_BSPEC = (
    ("EUNBG", None),
    ("EMISG", 10e-5),
    ("ESCALG", 1.5e-5),
    ("EBIASG", 1.5e-6),
    ("EWALKA", 4.1e-5),
    ("EMISA", 0.54e-4),
    ("ESCALA", 2e-6),
    ("EBIASA", 1.5e-3),
)
_GSPEC = (
    ("EUNBG", 4.83e-7),
    ("EMISG", 50e-6),
    ("ESCALG", 15e-5),
    ("EBIASG", 4.83e-6),
    ("EWALKA", 5.08e-5),
    ("EMISA", 4.85e-4),
    ("ESCALA", 3e-6),
    ("EBIASA", 9.81e-3),
)


def _spec_errors(rows):
    seed(0)
    drawn = {}
    for name, sig in rows:
        if sig is None:
            drawn[name] = np.zeros(3)
        else:
            drawn[name] = _gauss3_rtl(sig)
    return drawn


@pytest.mark.parametrize("mins,rows", ((2, _BSPEC), (3, _GSPEC)))
def test_sam6_mins_spec_errors(mins, rows):
    expected = _spec_errors(rows)
    veh = _sam6_missile(mins=mins)
    seed(0)
    _execute_module(veh, "ins", method="initialize")
    for name, _sig in rows:
        np.testing.assert_allclose(veh.store.get(name), expected[name], rtol=0, atol=0)


def test_sam6_mins2_execute_runs_error_model():
    veh = _sam6_missile(mins=2)
    seed(0)
    _execute_module(veh, "ins", method="initialize")
    _execute_module(veh, "ins")
    assert math.isfinite(veh.store.get("dvbec"))


def test_sam6_mterm_minus1():
    """C++ mterm==-1 miss is |STEL-SBEL| (line-of-sight), not the interpolated miss."""
    veh = _sam6_missile(mterm=-1)
    veh.store.set("mseek", 4)
    veh.store.set("alt", 1000.0)
    veh.store.set("hbe", 1000.0)
    veh.store.set("ip_sltrange", 1.0e9)
    veh.store.set("SBEL", (0.0, 0.0, 0.0))
    veh.store.set("SBMEL", (1.0, 0.0, 0.0))
    veh.store.set("STEL", (100.0, 0.0, 0.0))
    veh.store.set("VBEL", (0.0, 0.0, 0.0))
    veh.store.set("VTEL", (10.0, 0.0, 0.0))
    ctx = SimpleNamespace(
        int_step=0.01,
        vehicle_slot=0,
        combus=[SimpleNamespace(status=1)],
    )
    _execute_module(veh, "intercept", ctx=ctx)
    assert veh.store.get("miss") == pytest.approx(100.0)

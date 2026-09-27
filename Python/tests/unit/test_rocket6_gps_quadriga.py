import math

import numpy as np
import pytest

from cadac.constants import RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import StateStore
from cadac.math.wgs84 import GM, cad_in_geo84
from cadac.vehicles.round6.rocket6.gps import (
    LARGE,
    SV_INIT,
    Rocket6Gps,
    gps_quadriga,
    gps_sv_init,
    incl,
    rsi,
    wsi,
)

RTOL = 1e-12
ATOL = 1e-14

# C++ Hyper::def_gps order (ROCKET6 gps.cpp); unused slots 701, 712, 740-749 omitted.
DEFINED = (
    "mgps",
    "almanac_time",
    "del_rearth",
    "gdop",
    "gps_acqtime",
    "gps_step",
    "gps_epoch",
    "gps_acq",
    "ucfreq_noise",
    "ucbias_error",
    "ucfreq_error",
    "ucfreqm",
    "pr1_bias",
    "pr2_bias",
    "pr3_bias",
    "pr4_bias",
    "pr1_noise",
    "pr2_noise",
    "pr3_noise",
    "pr4_noise",
    "dr1_noise",
    "dr2_noise",
    "dr3_noise",
    "dr4_noise",
    "slotsum",
    "lon1",
    "lat1",
    "alt1",
    "lon2",
    "lat2",
    "alt2",
    "lon3",
    "lat3",
    "alt3",
    "lon4",
    "lat4",
    "alt4",
    "dum_alt",
    "uctime_cor",
    "ppos",
    "pvel",
    "pclockb",
    "pclockf",
    "qpos",
    "qvel",
    "qclockb",
    "qclockf",
    "rpos",
    "rvel",
    "factp",
    "factq",
    "factr",
    "SXH",
    "VXH",
    "CXH",
    "PP1",
    "PP2",
    "PP3",
    "PP4",
    "PP5",
    "PP6",
    "PP7",
    "PP8",
    "std_pos",
    "std_vel",
    "std_ucbias",
    "c2_pos_meas",
    "c2_vel_meas",
    "state_pos",
    "state_vel",
    "c2_range_err",
    "c2_delta_err",
)

ROLES = {
    "mgps": "data",
    "almanac_time": "data",
    "del_rearth": "data",
    "gdop": "diag",
    "gps_acqtime": "data",
    "gps_step": "data",
    "gps_epoch": "save",
    "gps_acq": "save",
    "ucfreq_noise": "data",
    "ucbias_error": "data",
    "ucfreq_error": "diag",
    "ucfreqm": "save",
    "pr1_bias": "data",
    "pr2_bias": "data",
    "pr3_bias": "data",
    "pr4_bias": "data",
    "pr1_noise": "data",
    "pr2_noise": "data",
    "pr3_noise": "data",
    "pr4_noise": "data",
    "dr1_noise": "data",
    "dr2_noise": "data",
    "dr3_noise": "data",
    "dr4_noise": "data",
    "slotsum": "save",
    "lon1": "diag",
    "lat1": "diag",
    "alt1": "diag",
    "lon2": "diag",
    "lat2": "diag",
    "alt2": "diag",
    "lon3": "diag",
    "lat3": "diag",
    "alt3": "diag",
    "lon4": "diag",
    "lat4": "diag",
    "alt4": "diag",
    "dum_alt": "diag",
    "uctime_cor": "data",
    "ppos": "data",
    "pvel": "data",
    "pclockb": "data",
    "pclockf": "data",
    "qpos": "data",
    "qvel": "data",
    "qclockb": "data",
    "qclockf": "data",
    "rpos": "data",
    "rvel": "data",
    "factp": "data",
    "factq": "data",
    "factr": "data",
    "SXH": "out",
    "VXH": "out",
    "CXH": "save",
    "PP1": "save",
    "PP2": "save",
    "PP3": "save",
    "PP4": "save",
    "PP5": "save",
    "PP6": "save",
    "PP7": "save",
    "PP8": "save",
    "std_pos": "diag",
    "std_vel": "diag",
    "std_ucbias": "diag",
    "c2_pos_meas": "diag",
    "c2_vel_meas": "diag",
    "state_pos": "diag",
    "state_vel": "diag",
    "c2_range_err": "diag",
    "c2_delta_err": "diag",
}

SCRN_PLOT = ("scrn", "plot")
PLOT = ("plot",)
OUTPUTS = {name: () for name in DEFINED}
OUTPUTS.update(
    {
        "ucbias_error": SCRN_PLOT,
        "ucfreq_error": SCRN_PLOT,
        "SXH": PLOT,
        "VXH": PLOT,
        "CXH": PLOT,
        "std_pos": PLOT,
        "std_vel": PLOT,
        "std_ucbias": PLOT,
        "c2_pos_meas": PLOT,
        "c2_vel_meas": PLOT,
        "state_pos": PLOT,
        "state_vel": PLOT,
        "c2_range_err": PLOT,
        "c2_delta_err": PLOT,
    }
)

INT_FIELDS = ("mgps", "gps_acq")
VEC_FIELDS = ("SXH", "VXH", "CXH")
MAT_FIELDS = ("PP1", "PP2", "PP3", "PP4", "PP5", "PP6", "PP7", "PP8")
NOT_DEFINED = (
    "time",
    "SBII",
    "VBII",
    "WBII",
    "SBIIC",
    "VBIIC",
    "WBICI",
)

# Vandenberg pad (input.asc) + GPS almanac from C++ gps_quadriga call site.
LONX = -120.49
LATX = 34.68
ALT = 100.0
ALMANAC_TIME = 80000.0
DEL_REARTH = 2317000.0
TIME = 0.0


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


def test_name_is_gps():
    # Break: class name token not "gps" (module bind).
    assert Rocket6Gps().name == "gps"


def test_define_registers_cpp_def_gps_fields():
    # Break: missing def_gps name, wrong role/type/outputs, or defining newton/INS names.
    vehicle = _Vehicle()
    Rocket6Gps().define(vehicle)
    store = vehicle.store
    for name in DEFINED:
        assert name in store.names(), name
        field = store.field(name)
        assert field.module == "gps"
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        if name in INT_FIELDS:
            assert field.type == "int"
            assert store.get(name) == 0
        elif name in VEC_FIELDS:
            assert field.type == "vec"
            np.testing.assert_array_equal(store.get(name), np.zeros(3))
        elif name in MAT_FIELDS:
            assert field.type == "mat"
            np.testing.assert_array_equal(store.get(name), np.zeros((3, 3)))
        else:
            assert field.type == "real"
            assert store.get(name) == 0.0
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_sv_init_yuma_week_787_slot1():
    # Break: Yuma 787 table not copied; slot #1 is 5.63, -1.600 in gps_sv_init.
    sv_data, rsi_out, wsi_out, incl_out = gps_sv_init()
    assert tuple(sv_data[0]) == (5.63, -1.600)
    np.testing.assert_allclose(SV_INIT[0], (5.63, -1.600), rtol=0.0, atol=0.0)
    assert sv_data.shape == (24, 2)
    assert SV_INIT.shape == (24, 2)
    assert rsi_out == 26560000
    assert rsi == 26560000
    assert incl_out == 0.95986
    assert incl == 0.95986
    want_wsi = math.sqrt(GM / rsi**3)
    assert wsi_out == pytest.approx(want_wsi, rel=RTOL, abs=ATOL)
    assert wsi == pytest.approx(want_wsi, rel=RTOL, abs=ATOL)


def test_vandenberg_quadriga_four_slots_finite_gdop():
    # Break: quadriga not ported, slots out of 1..24, or gdop left at LARGE/NaN.
    sbii = cad_in_geo84(LONX * RAD, LATX * RAD, ALT, TIME)
    sv_data, rsi_out, wsi_out, incl_out = gps_sv_init()
    ssii_quad, vsii_quad, gdop, mgps = gps_quadriga(
        sv_data,
        rsi_out,
        wsi_out,
        incl_out,
        ALMANAC_TIME,
        DEL_REARTH,
        TIME,
        sbii,
        3,
    )
    slots = [int(ssii_quad[i, 3]) for i in range(4)]
    assert len(slots) == 4
    assert all(1 <= slot <= 24 for slot in slots)
    assert len(set(slots)) == 4
    assert np.isfinite(gdop)
    assert gdop > 0.0
    assert gdop < LARGE
    assert mgps == 3
    assert ssii_quad.shape == (4, 4)
    assert vsii_quad.shape == (4, 3)
    assert np.all(np.isfinite(ssii_quad[:, :3]))
    assert np.all(np.isfinite(vsii_quad))


def test_fewer_than_four_visible_sets_mgps_one_without_cout(capsys):
    # Break: <4 visible leaves mgps=3, prints cout, or does not keep gdop=LARGE.
    sbii = cad_in_geo84(LONX * RAD, LATX * RAD, ALT, TIME)
    sv_data, rsi_out, wsi_out, incl_out = gps_sv_init()
    _ssii, _vsii, gdop, mgps = gps_quadriga(
        sv_data,
        rsi_out,
        wsi_out,
        incl_out,
        ALMANAC_TIME,
        20189000.0,
        TIME,
        sbii,
        3,
    )
    assert mgps == 1
    assert gdop == LARGE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_execute_may_pass_until_filter_task():
    # Break: execute missing (AttributeError) before Task 13 EKF.
    vehicle = _Vehicle()
    gps = Rocket6Gps()
    gps.define(vehicle)
    gps.initialize(vehicle, _ctx())
    gps.execute(vehicle, _ctx())
    assert vehicle.store.get("mgps") == 0
    assert vehicle.store.get("gdop") == 0.0

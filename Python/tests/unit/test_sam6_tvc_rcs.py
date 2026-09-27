from pathlib import Path

import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import StateStore
from cadac.vehicles.flat6.sam6.rcs import Sam6Rcs
from cadac.vehicles.flat6.sam6.tvc import Sam6Tvc

RTOL = 1e-12
ATOL = 1e-14

TVC_DEFINED = (
    "mtvc",
    "tvclimx",
    "dtvclimx",
    "wntvc",
    "zettvc",
    "pdynmc_gtvc36",
    "gtvc0",
    "parm",
    "FPB",
    "FMPB",
    "etax",
    "zetx",
    "etacx",
    "zetcx",
    "gtvc",
    "etasd",
    "zetad",
    "etas",
    "zeta",
    "detasd",
    "dzetad",
    "detas",
    "dzeta",
)
TVC_ROLES = {
    "mtvc": "data",
    "tvclimx": "data",
    "dtvclimx": "data",
    "wntvc": "data",
    "zettvc": "data",
    "pdynmc_gtvc36": "data",
    "gtvc0": "data",
    "parm": "data",
    "FPB": "out",
    "FMPB": "out",
    "etax": "diag",
    "zetx": "diag",
    "etacx": "diag",
    "zetcx": "diag",
    "gtvc": "out",
    "etasd": "state",
    "zetad": "state",
    "etas": "state",
    "zeta": "state",
    "detasd": "state",
    "dzetad": "state",
    "detas": "state",
    "dzeta": "state",
}
TVC_OUTPUTS = {
    "mtvc": (),
    "tvclimx": (),
    "dtvclimx": (),
    "wntvc": (),
    "zettvc": (),
    "pdynmc_gtvc36": (),
    "gtvc0": (),
    "parm": (),
    "FPB": (),
    "FMPB": (),
    "etax": ("plot",),
    "zetx": ("plot",),
    "etacx": (),
    "zetcx": (),
    "gtvc": (),
    "etasd": (),
    "zetad": (),
    "etas": (),
    "zeta": (),
    "detasd": (),
    "dzetad": (),
    "detas": (),
    "dzeta": (),
}
TVC_INT = ("mtvc",)
TVC_VEC = ("FPB", "FMPB")
TVC_NOT_DEFINED = (
    "time",
    "mprop",
    "maut",
    "thrust",
    "xcg",
    "dqcx",
    "drcx",
    "dqcx_rcs",
    "drcx_rcs",
    "pdynmc",
    "FAPB",
    "FMB",
    "FSPB",
    "FARCS",
    "FMRCS",
)

RCS_DEFINED = (
    "mrcs_moment",
    "mrcs_force",
    "dead_zone",
    "hysteresis",
    "rcs_tau",
    "roll_mom_max",
    "pitch_mom_max",
    "yaw_mom_max",
    "rcs_zeta",
    "rcs_freq",
    "rcs_arm",
    "roll_save",
    "pitch_save",
    "yaw_save",
    "FMRCS",
    "rcs_time",
    "rcs_isp",
    "rcs_fmass",
    "rate_gain_rcs",
    "phibdcomx",
    "thtbdcomx",
    "psibdcomx",
    "e_roll",
    "e_pitch",
    "e_yaw",
    "o_roll",
    "o_pitch",
    "o_yaw",
    "roll_count",
    "pitch_count",
    "yaw_count",
    "acc_gain",
    "rcs_thrust",
    "FARCS",
    "e_right",
    "e_down",
    "o_right",
    "o_down",
    "right_save",
    "down_save",
    "right_count",
    "down_count",
    "factdead_zone",
)
RCS_ROLES = {
    "mrcs_moment": "data",
    "mrcs_force": "data",
    "dead_zone": "data",
    "hysteresis": "data",
    "rcs_tau": "data",
    "roll_mom_max": "data",
    "pitch_mom_max": "data",
    "yaw_mom_max": "data",
    "rcs_zeta": "data",
    "rcs_freq": "data",
    "rcs_arm": "data",
    "roll_save": "save",
    "pitch_save": "save",
    "yaw_save": "save",
    "FMRCS": "out",
    "rcs_time": "save",
    "rcs_isp": "data",
    "rcs_fmass": "out",
    "rate_gain_rcs": "data",
    "phibdcomx": "data",
    "thtbdcomx": "data",
    "psibdcomx": "data",
    "e_roll": "diag",
    "e_pitch": "diag",
    "e_yaw": "diag",
    "o_roll": "save",
    "o_pitch": "save",
    "o_yaw": "save",
    "roll_count": "save",
    "pitch_count": "save",
    "yaw_count": "save",
    "acc_gain": "data",
    "rcs_thrust": "data",
    "FARCS": "out",
    "e_right": "diag",
    "e_down": "diag",
    "o_right": "save",
    "o_down": "save",
    "right_save": "save",
    "down_save": "save",
    "right_count": "save",
    "down_count": "save",
    "factdead_zone": "save",
}
RCS_OUTPUTS = {name: () for name in RCS_DEFINED}
RCS_INT = (
    "mrcs_moment",
    "mrcs_force",
    "o_roll",
    "o_pitch",
    "o_yaw",
    "roll_count",
    "pitch_count",
    "yaw_count",
    "o_right",
    "o_down",
    "right_count",
    "down_count",
)
RCS_VEC = ("FMRCS", "FARCS")
RCS_NOT_DEFINED = (
    "pdynmc",
    "xcg",
    "ai11",
    "ai33",
    "WBECB",
    "thtblcx",
    "FSPCB",
    "phiblcx",
    "alphacx",
    "betacx",
    "psibdcx",
    "ancomx",
    "alcomx",
    "UTBC",
    "alphacomx",
    "betacomx",
    "time",
    "FAPB",
    "FMB",
    "FPB",
    "FMPB",
)


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


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _assert_defined(store, defined, roles, outputs, int_fields, vec_fields, module):
    zeros3 = np.zeros(3)
    assert tuple(store.names()) == defined
    for name in defined:
        field = store.field(name)
        assert field.module == module, name
        assert field.role == roles[name], name
        assert field.outputs == outputs[name], name
        if name in int_fields:
            assert field.type == "int", name
            assert store.get(name) == 0, name
        elif name in vec_fields:
            assert field.type == "vec", name
            np.testing.assert_array_equal(store.get(name), zeros3)
            assert store.get(name).shape == (3,)
        else:
            assert field.type == "real", name
            assert _approx(store.get(name), 0.0), name


def _ready_tvc(*, mtvc=0):
    vehicle = _Vehicle()
    tvc = Sam6Tvc()
    tvc.define(vehicle)
    tvc.initialize(vehicle, _ctx())
    vehicle.store.set("mtvc", mtvc)
    return vehicle, tvc


def _ready_rcs(*, mrcs_moment=0, mrcs_force=0):
    vehicle = _Vehicle()
    rcs = Sam6Rcs()
    rcs.define(vehicle)
    rcs.initialize(vehicle, _ctx())
    vehicle.store.set("mrcs_moment", mrcs_moment)
    vehicle.store.set("mrcs_force", mrcs_force)
    return vehicle, rcs


def test_tvc_name_is_tvc():
    assert Sam6Tvc().name == "tvc"


def test_tvc_define_registers_cpp_fields():
    vehicle = _Vehicle()
    Sam6Tvc().define(vehicle)
    _assert_defined(
        vehicle.store,
        TVC_DEFINED,
        TVC_ROLES,
        TVC_OUTPUTS,
        TVC_INT,
        TVC_VEC,
        "tvc",
    )
    for name in TVC_NOT_DEFINED:
        assert name not in vehicle.store.names()


def test_tvc_initialize_is_pass():
    vehicle = _Vehicle()
    tvc = Sam6Tvc()
    tvc.define(vehicle)
    vehicle.store.set("FPB", (1.0, 2.0, 3.0))
    vehicle.store.set("gtvc", 4.0)
    before = {name: np.array(vehicle.store.get(name), copy=True) for name in TVC_VEC}
    before.update(
        {name: vehicle.store.get(name) for name in TVC_DEFINED if name not in TVC_VEC}
    )
    assert tvc.initialize(vehicle, _ctx()) is None
    for name in TVC_VEC:
        np.testing.assert_allclose(
            vehicle.store.get(name), before[name], rtol=RTOL, atol=ATOL
        )
    for name in TVC_DEFINED:
        if name in TVC_VEC:
            continue
        assert _approx(vehicle.store.get(name), before[name]), name


def test_mtvc_zero_execute_leaves_fpb_zero():
    vehicle, tvc = _ready_tvc(mtvc=0)
    tvc.execute(vehicle, _ctx())
    fpb = vehicle.store.get("FPB")
    fmpb = vehicle.store.get("FMPB")
    np.testing.assert_allclose(fpb, np.zeros(3), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(fmpb, np.zeros(3), rtol=RTOL, atol=ATOL)
    assert fpb is not fmpb
    assert _approx(vehicle.store.get("gtvc"), 0.0)
    assert _approx(vehicle.store.get("etax"), 0.0)
    assert _approx(vehicle.store.get("zetx"), 0.0)


def test_mtvc_zero_rewrites_dirty_fpb_to_zero():
    vehicle, tvc = _ready_tvc(mtvc=0)
    vehicle.store.set("FPB", (9.0, 8.0, 7.0))
    vehicle.store.set("FMPB", (6.0, 5.0, 4.0))
    tvc.execute(vehicle, _ctx())
    fpb = vehicle.store.get("FPB")
    fmpb = vehicle.store.get("FMPB")
    np.testing.assert_allclose(fpb, np.zeros(3), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(fmpb, np.zeros(3), rtol=RTOL, atol=ATOL)
    assert fpb is not fmpb
    fpb[0] = 1.0
    np.testing.assert_allclose(fmpb, np.zeros(3), rtol=RTOL, atol=ATOL)


def test_mtvc_one_raises():
    vehicle, tvc = _ready_tvc(mtvc=1)
    vehicle.store.set("FPB", (9.0, 8.0, 7.0))
    vehicle.store.set("FMPB", (6.0, 5.0, 4.0))
    with pytest.raises(ValueError):
        tvc.execute(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("FPB"), (9.0, 8.0, 7.0))
    np.testing.assert_array_equal(vehicle.store.get("FMPB"), (6.0, 5.0, 4.0))


@pytest.mark.parametrize("mtvc", [2, 3, -1, 4])
def test_mtvc_nonzero_raises(mtvc):
    vehicle, tvc = _ready_tvc(mtvc=mtvc)
    with pytest.raises(ValueError):
        tvc.execute(vehicle, _ctx())


def test_tvc_terminate_is_pass():
    vehicle, tvc = _ready_tvc(mtvc=0)
    assert tvc.terminate(vehicle, _ctx()) is None
    np.testing.assert_allclose(
        vehicle.store.get("FPB"), np.zeros(3), rtol=RTOL, atol=ATOL
    )


def test_tvc_no_flat6_or_plane_imports():
    import cadac.vehicles.flat6.sam6.tvc as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src


def test_rcs_name_is_rcs():
    assert Sam6Rcs().name == "rcs"


def test_rcs_define_registers_cpp_fields():
    vehicle = _Vehicle()
    Sam6Rcs().define(vehicle)
    _assert_defined(
        vehicle.store,
        RCS_DEFINED,
        RCS_ROLES,
        RCS_OUTPUTS,
        RCS_INT,
        RCS_VEC,
        "rcs",
    )
    for name in RCS_NOT_DEFINED:
        assert name not in vehicle.store.names()


def test_rcs_initialize_is_pass():
    vehicle = _Vehicle()
    rcs = Sam6Rcs()
    rcs.define(vehicle)
    vehicle.store.set("FARCS", (1.0, 2.0, 3.0))
    vehicle.store.set("rcs_fmass", 4.0)
    before_farcs = np.array(vehicle.store.get("FARCS"), copy=True)
    before_fmass = vehicle.store.get("rcs_fmass")
    assert rcs.initialize(vehicle, _ctx()) is None
    np.testing.assert_allclose(
        vehicle.store.get("FARCS"), before_farcs, rtol=RTOL, atol=ATOL
    )
    assert _approx(vehicle.store.get("rcs_fmass"), before_fmass)


def test_rcs_flags_zero_leave_farcs_zero():
    vehicle, rcs = _ready_rcs(mrcs_moment=0, mrcs_force=0)
    rcs.execute(vehicle, _ctx())
    farcs = vehicle.store.get("FARCS")
    fmrcs = vehicle.store.get("FMRCS")
    np.testing.assert_allclose(farcs, np.zeros(3), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(fmrcs, np.zeros(3), rtol=RTOL, atol=ATOL)
    assert farcs is not fmrcs
    assert _approx(vehicle.store.get("rcs_fmass"), 0.0)


def test_rcs_flags_zero_rewrite_dirty_vectors_to_zero():
    vehicle, rcs = _ready_rcs(mrcs_moment=0, mrcs_force=0)
    vehicle.store.set("FARCS", (11.0, 12.0, 13.0))
    vehicle.store.set("FMRCS", (21.0, 22.0, 23.0))
    rcs.execute(vehicle, _ctx())
    farcs = vehicle.store.get("FARCS")
    fmrcs = vehicle.store.get("FMRCS")
    np.testing.assert_allclose(farcs, np.zeros(3), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(fmrcs, np.zeros(3), rtol=RTOL, atol=ATOL)
    assert farcs is not fmrcs
    farcs[0] = 1.0
    np.testing.assert_allclose(fmrcs, np.zeros(3), rtol=RTOL, atol=ATOL)


def test_mrcs_force_one_raises():
    vehicle, rcs = _ready_rcs(mrcs_moment=0, mrcs_force=1)
    vehicle.store.set("FARCS", (11.0, 12.0, 13.0))
    vehicle.store.set("FMRCS", (21.0, 22.0, 23.0))
    with pytest.raises(ValueError):
        rcs.execute(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("FARCS"), (11.0, 12.0, 13.0))
    np.testing.assert_array_equal(vehicle.store.get("FMRCS"), (21.0, 22.0, 23.0))


@pytest.mark.parametrize("mrcs_moment", [1, 10, 11, 20, 24])
def test_mrcs_moment_nonzero_raises(mrcs_moment):
    vehicle, rcs = _ready_rcs(mrcs_moment=mrcs_moment, mrcs_force=0)
    with pytest.raises(ValueError):
        rcs.execute(vehicle, _ctx())


@pytest.mark.parametrize("mrcs_force", [2, -1, 3])
def test_mrcs_force_other_nonzero_raises(mrcs_force):
    vehicle, rcs = _ready_rcs(mrcs_moment=0, mrcs_force=mrcs_force)
    with pytest.raises(ValueError):
        rcs.execute(vehicle, _ctx())


def test_rcs_terminate_is_pass():
    vehicle, rcs = _ready_rcs()
    assert rcs.terminate(vehicle, _ctx()) is None
    np.testing.assert_allclose(
        vehicle.store.get("FARCS"), np.zeros(3), rtol=RTOL, atol=ATOL
    )


def test_rcs_no_flat6_or_plane_imports():
    import cadac.vehicles.flat6.sam6.rcs as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src

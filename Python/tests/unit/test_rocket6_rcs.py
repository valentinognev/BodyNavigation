import numpy as np
import pytest

from cadac.constants import AGRAV, DEG
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round6.rocket6.rcs import Rocket6Rcs, rcs_prop, rcs_schmitt

RTOL = 1e-12
ATOL = 1e-14

# insertion RCS data (input.asc)
MRCS_MOMENT = 21
DEAD_ZONE = 0.4
HYSTERESIS = 0.1
RCS_TAU = 1.0
ROLL_MOM_MAX = 100.0
PITCH_MOM_MAX = 200000.0
YAW_MOM_MAX = 200000.0
THTBDCOMX = 80.0
PSIBCOMX = -83.0
DT = 0.001

# proportional / side-force fixtures (C++ rcs_type==1, mrcs_force 1|2)
RCS_ZETA = 0.7
RCS_FREQ = 10.0
ACC_GAIN = 5.0
SIDE_FORCE_MAX = 50.0
IBBB_DIAG = (2.0, 40.0, 45.0)
IBBB = np.diag(IBBB_DIAG)

# C++ Hyper::rcs_schmitt with CADAC sign(x<0 → -1 else +1).
# Columns: input_new, input, dead_zone, hysteresis, output
SCHMITT_TABLE = (
    (0.0, 0.0, 0.4, 0.1, 0),
    (0.1, 0.0, 0.4, 0.1, 0),
    (0.3, 0.2, 0.4, 0.1, 0),
    (0.3, 0.25, 0.4, 0.1, 1),
    (0.5, 0.4, 0.4, 0.1, 1),
    (0.2, 0.3, 0.4, 0.1, 1),
    (0.1, 0.16, 0.4, 0.1, 1),
    (0.1, 0.14, 0.4, 0.1, 0),
    (-0.3, -0.2, 0.4, 0.1, 0),
    (-0.3, -0.25, 0.4, 0.1, -1),
    (-0.5, -0.4, 0.4, 0.1, -1),
    (-0.2, -0.3, 0.4, 0.1, -1),
    (-0.1, -0.14, 0.4, 0.1, 0),
    (1.0, 0.0, 0.0, 0.0, 1),
    (-1.0, 0.0, 0.0, 0.0, 1),
    (1.0, 0.1, 0.0, 0.0, 1),
    (-1.0, -0.1, 0.0, 0.0, -1),
    (0.3, 0.25, 0.4, 0.0, 1),
    (0.3, 0.15, 0.4, 0.0, 0),
    (0.1, 0.06, 0.0, 0.1, 1),
    (0.1, 0.04, 0.0, 0.1, 0),
)

DEFINED = (
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
    "roll_save",
    "pitch_save",
    "yaw_save",
    "FMRCS",
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
    "side_force_max",
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

ROLES = {
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
    "roll_save": "save",
    "pitch_save": "save",
    "yaw_save": "save",
    "FMRCS": "out",
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
    "side_force_max": "data",
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

OUTPUTS = {
    "mrcs_moment": (),
    "mrcs_force": (),
    "dead_zone": ("plot",),
    "hysteresis": (),
    "rcs_tau": (),
    "roll_mom_max": (),
    "pitch_mom_max": (),
    "yaw_mom_max": (),
    "rcs_zeta": (),
    "rcs_freq": (),
    "roll_save": (),
    "pitch_save": (),
    "yaw_save": (),
    "FMRCS": (),
    "phibdcomx": (),
    "thtbdcomx": (),
    "psibdcomx": (),
    "e_roll": (),
    "e_pitch": (),
    "e_yaw": (),
    "o_roll": (),
    "o_pitch": (),
    "o_yaw": (),
    "roll_count": ("scrn", "plot"),
    "pitch_count": ("scrn", "plot"),
    "yaw_count": ("scrn", "plot"),
    "acc_gain": (),
    "side_force_max": (),
    "FARCS": (),
    "e_right": (),
    "e_down": (),
    "o_right": (),
    "o_down": (),
    "right_save": (),
    "down_save": (),
    "right_count": (),
    "down_count": (),
    "factdead_zone": (),
}

INT_FIELDS = (
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
VEC_FIELDS = ("FMRCS", "FARCS")
NOT_DEFINED = (
    "phibdcx",
    "thtbdcx",
    "psibdcx",
    "UTBC",
    "ppcx",
    "qqcx",
    "rrcx",
    "alphacx",
    "betacx",
    "alphacomx",
    "betacomx",
    "FSPCB",
    "IBBB",
    "aycomx",
    "azcomx",
    "pdynmc",
    "time",
    "minit",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def _ctx(dt=DT):
    return SimContext(
        sim_time=0.0,
        int_step=dt,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _plant_externals(
    store,
    *,
    ppcx=0.0,
    qqcx=0.0,
    rrcx=0.0,
    phibdcx=0.0,
    thtbdcx=0.0,
    psibdcx=0.0,
    utbc=(0.0, 0.0, 0.0),
    alphacx=0.0,
    betacx=0.0,
    alphacomx=0.0,
    betacomx=0.0,
    ibbb=None,
    fspcb=(0.0, 0.0, 0.0),
    aycomx=0.0,
    azcomx=0.0,
):
    store.define(Field("ppcx", ppcx, "real", "out", "ins"))
    store.define(Field("qqcx", qqcx, "real", "out", "ins"))
    store.define(Field("rrcx", rrcx, "real", "out", "ins"))
    store.define(Field("phibdcx", phibdcx, "real", "out", "ins"))
    store.define(Field("thtbdcx", thtbdcx, "real", "out", "ins"))
    store.define(Field("psibdcx", psibdcx, "real", "out", "ins"))
    store.define(Field("UTBC", utbc, "vec", "out", "guidance"))
    store.define(Field("alphacx", alphacx, "real", "out", "ins"))
    store.define(Field("betacx", betacx, "real", "out", "ins"))
    store.define(Field("alphacomx", alphacomx, "real", "out", "guidance"))
    store.define(Field("betacomx", betacomx, "real", "out", "guidance"))
    if ibbb is None:
        ibbb = IBBB
    store.define(Field("IBBB", ibbb, "mat", "out", "propulsion"))
    store.define(Field("FSPCB", fspcb, "vec", "out", "ins"))
    store.define(Field("aycomx", aycomx, "real", "out", "guidance"))
    store.define(Field("azcomx", azcomx, "real", "out", "guidance"))


def _ready(
    *,
    mrcs_moment=MRCS_MOMENT,
    mrcs_force=0,
    dead_zone=DEAD_ZONE,
    hysteresis=HYSTERESIS,
    rcs_tau=RCS_TAU,
    roll_mom_max=ROLL_MOM_MAX,
    pitch_mom_max=PITCH_MOM_MAX,
    yaw_mom_max=YAW_MOM_MAX,
    rcs_zeta=RCS_ZETA,
    rcs_freq=RCS_FREQ,
    acc_gain=ACC_GAIN,
    side_force_max=SIDE_FORCE_MAX,
    phibdcomx=0.0,
    thtbdcomx=THTBDCOMX,
    psibdcomx=PSIBCOMX,
    ppcx=0.0,
    qqcx=0.0,
    rrcx=0.0,
    phibdcx=0.0,
    thtbdcx=0.0,
    psibdcx=0.0,
    utbc=(0.0, 0.0, 0.0),
    alphacx=0.0,
    betacx=0.0,
    alphacomx=0.0,
    betacomx=0.0,
    ibbb=None,
    fspcb=(0.0, 0.0, 0.0),
    aycomx=0.0,
    azcomx=0.0,
    **states,
):
    vehicle = _Vehicle()
    rcs = Rocket6Rcs()
    rcs.define(vehicle)
    _plant_externals(
        vehicle.store,
        ppcx=ppcx,
        qqcx=qqcx,
        rrcx=rrcx,
        phibdcx=phibdcx,
        thtbdcx=thtbdcx,
        psibdcx=psibdcx,
        utbc=utbc,
        alphacx=alphacx,
        betacx=betacx,
        alphacomx=alphacomx,
        betacomx=betacomx,
        ibbb=ibbb,
        fspcb=fspcb,
        aycomx=aycomx,
        azcomx=azcomx,
    )
    store = vehicle.store
    store.set("mrcs_moment", mrcs_moment)
    store.set("mrcs_force", mrcs_force)
    store.set("dead_zone", dead_zone)
    store.set("hysteresis", hysteresis)
    store.set("rcs_tau", rcs_tau)
    store.set("roll_mom_max", roll_mom_max)
    store.set("pitch_mom_max", pitch_mom_max)
    store.set("yaw_mom_max", yaw_mom_max)
    store.set("rcs_zeta", rcs_zeta)
    store.set("rcs_freq", rcs_freq)
    store.set("acc_gain", acc_gain)
    store.set("side_force_max", side_force_max)
    store.set("phibdcomx", phibdcomx)
    store.set("thtbdcomx", thtbdcomx)
    store.set("psibdcomx", psibdcomx)
    for name, value in states.items():
        store.set(name, value)
    rcs.initialize(vehicle, _ctx())
    return vehicle, rcs


def test_name_is_rcs():
    assert Rocket6Rcs().name == "rcs"


def test_define_registers_cpp_fields_not_externals():
    vehicle = _Vehicle()
    Rocket6Rcs().define(vehicle)
    store = vehicle.store
    for name in DEFINED:
        assert name in store.names(), name
        field = store.field(name)
        assert field.module == "rcs"
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        if name in INT_FIELDS:
            assert field.type == "int"
            assert store.get(name) == 0
        elif name in VEC_FIELDS:
            assert field.type == "vec"
            assert np.array_equal(store.get(name), np.zeros(3))
        else:
            assert field.type == "real"
            assert store.get(name) == 0.0
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_initialize_is_noop():
    vehicle = _Vehicle()
    rcs = Rocket6Rcs()
    rcs.define(vehicle)
    store = vehicle.store
    store.set("mrcs_moment", 21)
    store.set("roll_mom_max", ROLL_MOM_MAX)
    rcs.initialize(vehicle, _ctx())
    assert store.get("mrcs_moment") == 21
    assert store.get("roll_mom_max") == ROLL_MOM_MAX
    np.testing.assert_array_equal(store.get("FMRCS"), np.zeros(3))
    np.testing.assert_array_equal(store.get("FARCS"), np.zeros(3))
    assert store.get("o_roll") == 0
    assert store.get("e_roll") == 0.0


@pytest.mark.parametrize(
    "input_new, previous, dead_zone, hysteresis, output",
    SCHMITT_TABLE,
)
def test_rcs_schmitt_matches_cpp_table(input_new, previous, dead_zone, hysteresis, output):
    # Break: np.sign(0)==0, compare input_new instead of input, or invert hysteresis.
    assert rcs_schmitt(input_new, previous, dead_zone, hysteresis) == output


def test_rcs_schmitt_cadac_sign_zero_is_plus_one_not_numpy_sign():
    # input==0 → CADAC side=+1, trigger=0 when dz=hy=0 → output +1.
    # np.sign(0)==0 would take the else branch → 0.
    assert _sign(0.0) == 1
    assert np.sign(0.0) == 0.0
    assert rcs_schmitt(-1.0, 0.0, 0.0, 0.0) == 1
    assert rcs_schmitt(-1.0, 0.0, 0.0, 0.0) != int(np.sign(0.0))


def test_mrcs_moment_0_no_moments():
    vehicle, rcs = _ready(mrcs_moment=0, thtbdcomx=THTBDCOMX, pitch_save=80.0)
    rcs.execute(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("FMRCS"), np.zeros(3))
    np.testing.assert_array_equal(vehicle.store.get("FARCS"), np.zeros(3))


def test_mrcs_moment_21_finite_fmrcs():
    # Previous Euler errors sit outside the dead zone so Schmitt fires this step.
    vehicle, rcs = _ready(
        mrcs_moment=21,
        phibdcomx=0.0,
        thtbdcomx=THTBDCOMX,
        psibdcomx=PSIBCOMX,
        roll_save=0.0,
        pitch_save=THTBDCOMX,
        yaw_save=PSIBCOMX,
    )
    rcs.execute(vehicle, _ctx())
    fmrcs = vehicle.store.get("FMRCS")
    assert np.all(np.isfinite(fmrcs))
    assert _approx(fmrcs[0], 0.0)
    assert _approx(fmrcs[1], PITCH_MOM_MAX)
    assert _approx(fmrcs[2], -YAW_MOM_MAX)
    assert fmrcs[1] != 0.0
    assert fmrcs[2] != 0.0
    np.testing.assert_array_equal(vehicle.store.get("FARCS"), np.zeros(3))
    assert _approx(vehicle.store.get("e_pitch"), THTBDCOMX)
    assert _approx(vehicle.store.get("e_yaw"), PSIBCOMX)
    assert vehicle.store.get("o_pitch") == 1
    assert vehicle.store.get("o_yaw") == -1


def test_rcs_prop_limits():
    assert _approx(rcs_prop(50.0, 100.0), 50.0)
    assert _approx(rcs_prop(150.0, 100.0), 100.0)
    assert _approx(rcs_prop(-150.0, 100.0), -100.0)


def test_mrcs_moment_11_prop_euler_writes_fmrcs():
    # C++ rcs_type==1 / rcs_mode==1: gain from IBBB diag; FMRCS = rcs_prop(e, mom_max).
    phibdcomx = 5.0
    thtbdcomx = 8.0
    psibdcomx = -6.0
    phibdcx = 0.0
    thtbdcx = 0.0
    psibdcx = 0.0
    ppcx = qqcx = rrcx = 0.0
    vehicle, rcs = _ready(
        mrcs_moment=11,
        mrcs_force=0,
        phibdcomx=phibdcomx,
        thtbdcomx=thtbdcomx,
        psibdcomx=psibdcomx,
        phibdcx=phibdcx,
        thtbdcx=thtbdcx,
        psibdcx=psibdcx,
        ppcx=ppcx,
        qqcx=qqcx,
        rrcx=rrcx,
        roll_mom_max=ROLL_MOM_MAX,
        pitch_mom_max=200.0,
        yaw_mom_max=150.0,
    )
    rcs.execute(vehicle, _ctx())
    store = vehicle.store
    rgain_roll = 2.0 * RCS_ZETA * RCS_FREQ * IBBB_DIAG[0]
    rgain_pitch = 2.0 * RCS_ZETA * RCS_FREQ * IBBB_DIAG[1]
    rgain_yaw = 2.0 * RCS_ZETA * RCS_FREQ * IBBB_DIAG[2]
    pgain = RCS_FREQ / (2.0 * RCS_ZETA)
    e_roll = rgain_roll * (pgain * (phibdcomx - phibdcx) - ppcx)
    e_pitch = rgain_pitch * (pgain * (thtbdcomx - thtbdcx) - qqcx)
    e_yaw = rgain_yaw * (pgain * (psibdcomx - psibdcx) - rrcx)
    want = np.array(
        [
            rcs_prop(e_roll, ROLL_MOM_MAX),
            rcs_prop(e_pitch, 200.0),
            rcs_prop(e_yaw, 150.0),
        ]
    )
    fmrcs = store.get("FMRCS")
    assert np.all(np.isfinite(fmrcs))
    np.testing.assert_allclose(fmrcs, want, rtol=RTOL, atol=ATOL)
    assert not np.allclose(fmrcs, 0.0)
    np.testing.assert_allclose(store.get("FARCS"), np.zeros(3), rtol=RTOL, atol=ATOL)
    assert _approx(store.get("e_roll"), e_roll)
    assert _approx(store.get("e_pitch"), e_pitch)
    assert _approx(store.get("e_yaw"), e_yaw)


def test_mrcs_moment_12_prop_utbc_writes_fmrcs():
    # C++ rcs_mode==2: e_pitch/yaw from UTBC*DEG with prop gains.
    utbc = (0.0, 0.1, -0.2)
    qqcx = 1.0
    rrcx = -0.5
    vehicle, rcs = _ready(
        mrcs_moment=12,
        mrcs_force=0,
        utbc=utbc,
        qqcx=qqcx,
        rrcx=rrcx,
        thtbdcomx=THTBDCOMX,
        psibdcomx=PSIBCOMX,
        pitch_mom_max=200.0,
        yaw_mom_max=150.0,
    )
    rcs.execute(vehicle, _ctx())
    store = vehicle.store
    rgain_pitch = 2.0 * RCS_ZETA * RCS_FREQ * IBBB_DIAG[1]
    rgain_yaw = 2.0 * RCS_ZETA * RCS_FREQ * IBBB_DIAG[2]
    pgain = RCS_FREQ / (2.0 * RCS_ZETA)
    e_pitch = rgain_pitch * (pgain * (-utbc[2]) * DEG - qqcx)
    e_yaw = rgain_yaw * (pgain * (utbc[1]) * DEG - rrcx)
    assert _approx(store.get("e_pitch"), e_pitch)
    assert _approx(store.get("e_yaw"), e_yaw)
    assert store.get("e_pitch") != pytest.approx(THTBDCOMX, rel=RTOL, abs=ATOL)
    fmrcs = store.get("FMRCS")
    assert _approx(fmrcs[1], rcs_prop(e_pitch, 200.0))
    assert _approx(fmrcs[2], rcs_prop(e_yaw, 150.0))


def test_mrcs_moment_10_prop_roll_only():
    # Mode 0: pitch/yaw errors stay 0; roll still proportional.
    vehicle, rcs = _ready(
        mrcs_moment=10,
        phibdcomx=5.0,
        phibdcx=0.0,
        ppcx=0.0,
        thtbdcomx=THTBDCOMX,
        psibdcomx=PSIBCOMX,
        pitch_mom_max=200.0,
        yaw_mom_max=150.0,
    )
    rcs.execute(vehicle, _ctx())
    store = vehicle.store
    rgain_roll = 2.0 * RCS_ZETA * RCS_FREQ * IBBB_DIAG[0]
    pgain = RCS_FREQ / (2.0 * RCS_ZETA)
    e_roll = rgain_roll * (pgain * 5.0 - 0.0)
    assert _approx(store.get("e_roll"), e_roll)
    assert store.get("e_pitch") == 0.0
    assert store.get("e_yaw") == 0.0
    fmrcs = store.get("FMRCS")
    assert _approx(fmrcs[0], rcs_prop(e_roll, ROLL_MOM_MAX))
    assert _approx(fmrcs[1], 0.0)
    assert _approx(fmrcs[2], 0.0)


def test_mrcs_moment_13_prop_roll_only_no_incidence_branch():
    # Type 1 has no mode-3 incidence path in C++; pitch/yaw stay 0.
    vehicle, rcs = _ready(
        mrcs_moment=13,
        phibdcomx=4.0,
        alphacomx=10.0,
        betacomx=3.0,
        pitch_mom_max=200.0,
        yaw_mom_max=150.0,
    )
    rcs.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("e_pitch") == 0.0
    assert store.get("e_yaw") == 0.0
    assert store.get("FMRCS")[0] != 0.0
    assert _approx(store.get("FMRCS")[1], 0.0)
    assert _approx(store.get("FMRCS")[2], 0.0)


def test_mrcs_force_1_prop_writes_farcs():
    # C++ mrcs_force==1: e_right/down = acc_gain*(a*com*AGRAV - fsp); FARCS = rcs_prop.
    aycomx = 2.0
    azcomx = 1.5
    fspcb = (0.0, 1.0, -2.0)
    vehicle, rcs = _ready(
        mrcs_moment=0,
        mrcs_force=1,
        aycomx=aycomx,
        azcomx=azcomx,
        fspcb=fspcb,
    )
    rcs.execute(vehicle, _ctx())
    store = vehicle.store
    e_right = ACC_GAIN * (aycomx * AGRAV - fspcb[1])
    e_down = ACC_GAIN * (azcomx * AGRAV - fspcb[2])
    want = np.array(
        [0.0, rcs_prop(e_right, SIDE_FORCE_MAX), rcs_prop(e_down, SIDE_FORCE_MAX)]
    )
    farcs = store.get("FARCS")
    np.testing.assert_allclose(farcs, want, rtol=RTOL, atol=ATOL)
    assert not np.allclose(farcs[1:], 0.0)
    assert _approx(store.get("e_right"), e_right)
    assert _approx(store.get("e_down"), e_down)
    np.testing.assert_allclose(store.get("FMRCS"), np.zeros(3), rtol=RTOL, atol=ATOL)


def test_mrcs_force_2_schmitt_writes_farcs():
    # C++ mrcs_force==2: Schmitt on accel errors; FARCS = o * side_force_max (no parasitic FMRCS).
    aycomx = 3.0
    azcomx = -2.0
    fspcb = (0.0, 0.0, 0.0)
    e_right = ACC_GAIN * (aycomx * AGRAV - 0.0)
    e_down = ACC_GAIN * (azcomx * AGRAV - 0.0)
    vehicle, rcs = _ready(
        mrcs_moment=0,
        mrcs_force=2,
        aycomx=aycomx,
        azcomx=azcomx,
        fspcb=fspcb,
        right_save=e_right,
        down_save=e_down,
        o_right=0,
        o_down=0,
        right_count=0,
        down_count=0,
    )
    rcs.execute(vehicle, _ctx())
    store = vehicle.store
    assert store.get("o_right") == 1
    assert store.get("o_down") == -1
    farcs = store.get("FARCS")
    assert _approx(farcs[0], 0.0)
    assert _approx(farcs[1], SIDE_FORCE_MAX)
    assert _approx(farcs[2], -SIDE_FORCE_MAX)
    assert store.get("right_count") == 1
    assert store.get("down_count") == 1
    assert _approx(store.get("right_save"), e_right)
    assert _approx(store.get("down_save"), e_down)
    np.testing.assert_allclose(store.get("FMRCS"), np.zeros(3), rtol=RTOL, atol=ATOL)


def test_other_mrcs_moment_raises():
    for moment in (1, 2, 30, -1):
        vehicle, rcs = _ready(mrcs_moment=moment)
        with pytest.raises(ValueError):
            rcs.execute(vehicle, _ctx())


def test_other_mrcs_force_raises():
    for force in (3, -1):
        vehicle, rcs = _ready(mrcs_moment=0, mrcs_force=force)
        with pytest.raises(ValueError):
            rcs.execute(vehicle, _ctx())


def test_rocket6_mrcs_moment_23_schmitt():
    # C++ rcs_mode==3: e_pitch=alphacomx-(rcs_tau*qqcx+alphacx);
    # e_yaw=-betacomx-(rcs_tau*rrcx-betacx). Ballistic-case alphacomx=10.
    alphacomx = 10.0
    betacomx = 0.0
    alphacx = 0.0
    betacx = 0.0
    qqcx = 0.0
    rrcx = 0.0
    e_pitch = alphacomx - (RCS_TAU * qqcx + alphacx)
    e_yaw = -betacomx - (RCS_TAU * rrcx - betacx)
    vehicle, rcs = _ready(
        mrcs_moment=23,
        alphacomx=alphacomx,
        betacomx=betacomx,
        alphacx=alphacx,
        betacx=betacx,
        qqcx=qqcx,
        rrcx=rrcx,
        thtbdcomx=THTBDCOMX,
        psibdcomx=PSIBCOMX,
        pitch_save=e_pitch,
        yaw_save=e_yaw,
    )
    rcs.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("e_pitch"), e_pitch)
    assert _approx(store.get("e_yaw"), e_yaw)
    assert store.get("e_pitch") != pytest.approx(THTBDCOMX, rel=RTOL, abs=ATOL)
    fmrcs = store.get("FMRCS")
    assert np.all(np.isfinite(fmrcs))
    assert _approx(fmrcs[1], PITCH_MOM_MAX * store.get("o_pitch"))
    assert _approx(fmrcs[2], YAW_MOM_MAX * store.get("o_yaw"))
    assert store.get("o_pitch") == 1
    assert store.get("o_yaw") == 0


def test_mrcs_moment_20_roll_only_ignores_euler_commands():
    # pitch_save in the hysteresis band: trend from e_pitch=0 (mode 0) is
    # decreasing → on; Euler mode 1 with thtbdcomx=80 would be increasing → off.
    vehicle, rcs = _ready(
        mrcs_moment=20,
        phibdcomx=5.0,
        thtbdcomx=THTBDCOMX,
        psibdcomx=PSIBCOMX,
        roll_save=5.0,
        pitch_save=0.20,
        yaw_save=0.20,
    )
    rcs.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("e_roll"), 5.0)
    assert store.get("e_pitch") == 0.0
    assert store.get("e_yaw") == 0.0
    assert store.get("o_roll") == 1
    assert store.get("o_pitch") == 1
    assert store.get("o_yaw") == 1
    fmrcs = store.get("FMRCS")
    assert _approx(fmrcs[0], ROLL_MOM_MAX)
    assert _approx(fmrcs[1], PITCH_MOM_MAX)
    assert _approx(fmrcs[2], YAW_MOM_MAX)


def test_mrcs_moment_22_utbc_vector_errors():
    utbc = (0.0, 0.1, -0.2)
    vehicle, rcs = _ready(
        mrcs_moment=22,
        qqcx=0.0,
        rrcx=0.0,
        utbc=utbc,
        thtbdcomx=THTBDCOMX,
        psibdcomx=PSIBCOMX,
        pitch_save=utbc[2] * 0.0 + 0.2 * DEG,
        yaw_save=0.1 * DEG,
    )
    rcs.execute(vehicle, _ctx())
    store = vehicle.store
    e_pitch = -RCS_TAU * 0.0 - utbc[2] * DEG
    e_yaw = -RCS_TAU * 0.0 + utbc[1] * DEG
    assert _approx(store.get("e_pitch"), e_pitch)
    assert _approx(store.get("e_yaw"), e_yaw)
    assert store.get("e_pitch") != pytest.approx(THTBDCOMX, rel=RTOL, abs=ATOL)
    fmrcs = store.get("FMRCS")
    assert np.all(np.isfinite(fmrcs))
    assert _approx(fmrcs[1], PITCH_MOM_MAX * store.get("o_pitch"))
    assert _approx(fmrcs[2], YAW_MOM_MAX * store.get("o_yaw"))
    assert store.get("o_pitch") == 1
    assert store.get("o_yaw") == 1


def test_rate_damping_in_error():
    vehicle, rcs = _ready(
        mrcs_moment=21,
        thtbdcomx=THTBDCOMX,
        thtbdcx=10.0,
        qqcx=5.0,
        rcs_tau=RCS_TAU,
        pitch_save=THTBDCOMX - (RCS_TAU * 5.0 + 10.0),
    )
    rcs.execute(vehicle, _ctx())
    assert _approx(vehicle.store.get("e_pitch"), THTBDCOMX - (RCS_TAU * 5.0 + 10.0))


def test_count_increments_when_schmitt_output_changes():
    vehicle, rcs = _ready(
        mrcs_moment=21,
        thtbdcomx=THTBDCOMX,
        psibdcomx=PSIBCOMX,
        pitch_save=THTBDCOMX,
        yaw_save=PSIBCOMX,
        o_pitch=0,
        o_yaw=0,
        pitch_count=0,
        yaw_count=0,
    )
    rcs.execute(vehicle, _ctx())
    assert vehicle.store.get("pitch_count") == 1
    assert vehicle.store.get("yaw_count") == 1
    rcs.execute(vehicle, _ctx())
    assert vehicle.store.get("pitch_count") == 1
    assert vehicle.store.get("yaw_count") == 1


def test_saves_updated_to_current_errors():
    vehicle, rcs = _ready(
        mrcs_moment=21,
        thtbdcomx=THTBDCOMX,
        psibdcomx=PSIBCOMX,
        pitch_save=0.0,
        yaw_save=0.0,
    )
    rcs.execute(vehicle, _ctx())
    assert _approx(vehicle.store.get("pitch_save"), THTBDCOMX)
    assert _approx(vehicle.store.get("yaw_save"), PSIBCOMX)
    assert _approx(vehicle.store.get("e_pitch"), THTBDCOMX)


def test_farcs_stays_zero_when_force_disabled():
    vehicle, rcs = _ready(mrcs_moment=21, pitch_save=THTBDCOMX)
    rcs.execute(vehicle, _ctx())
    np.testing.assert_array_equal(vehicle.store.get("FARCS"), np.zeros(3))


def test_terminate_exists_and_is_pass():
    vehicle, rcs = _ready(mrcs_moment=0)
    assert rcs.terminate(vehicle, _ctx()) is None

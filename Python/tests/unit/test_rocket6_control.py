import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.rocket6.control import Rocket6Control

RTOL = 1e-12
ATOL = 1e-14

# insertion control data (input.asc) plus frozen aero/INS plants
MAUT = 53
DELIMX = 10.0
DRLIMX = 10.0
ZACLP = 1.0
ZACLY = 1.0
FACTWACLP = 0.5
FACTWACLY = 0.5
ANCOMX = -0.15
ALCOMX = 0.2
DT = 0.001
MPROP = 3
GNMAX = 5.0
GYMAX = 5.0
PDYNMC = 5000.0
DLA = 10.0
DMA = -2.0
DMQ = -0.5
DMDE = -20.0
DYB = -10.0
DNB = 2.0
DNR = -0.5
DNDR = -20.0
DVBEC = 170.0
DVBE = 200.0
QQCX = 1.0
RRCX = 0.5
FSPCB = (0.0, 1.0, -9.8)

# C++ Hyper::control_normal_accel / control_yaw_accel, one 0.001 s step,
# frozen plants above, states start at 0. Hand-checked vs CADAC formulas.
CPP_DELECX = -5.5925776608273114
CPP_DELRCX = -0.5695111878760382
CPP_WACLP = 0.03749999999999999
CPP_PACLP = 0.4749999999999999
CPP_GAINFP = (0.009959307417820067, 0.00044117647058823856, -3.3398437499999977e-06)
CPP_ZZD = -11.271013167500001
CPP_ZZ = -0.005635506583750001
CPP_GAINFY = (0.00993984375, 3.469446951953614e-18, -3.3398437499999977e-06)
CPP_YYD = 0.9613508900000001
CPP_YY = 0.000480675445
CPP_DELECX_STEP2 = -5.592575504018053
CPP_DELRCX_STEP2 = -0.5695113718391361
CPP_DELECX_GNMAX = -5.592577754657824
CPP_DELRCX_GYMAX = -0.5695110471302697
CPP_DELRCX_IF_DVBEC = -0.5710253918944563

DEFINED = (
    "maut",
    "mfreeze",
    "waclp",
    "zaclp",
    "paclp",
    "delimx",
    "drlimx",
    "yyd",
    "yy",
    "zzd",
    "zz",
    "delecx",
    "delrcx",
    "alcomx_actual",
    "ancomx_actual",
    "GAINFP",
    "gainl",
    "gkp",
    "gkphi",
    "isetc2",
    "wacly",
    "zacly",
    "pacly",
    "GAINFY",
    "factwaclp",
    "factwacly",
    "alcomx",
    "ancomx",
)

ROLES = {
    "maut": "data",
    "mfreeze": "data",
    "waclp": "data",
    "zaclp": "data",
    "paclp": "data",
    "delimx": "data",
    "drlimx": "data",
    "yyd": "state",
    "yy": "state",
    "zzd": "state",
    "zz": "state",
    "delecx": "out",
    "delrcx": "out",
    "alcomx_actual": "diag",
    "ancomx_actual": "diag",
    "GAINFP": "diag",
    "gainl": "data",
    "gkp": "diag",
    "gkphi": "diag",
    "isetc2": "init",
    "wacly": "data",
    "zacly": "data",
    "pacly": "data",
    "GAINFY": "diag",
    "factwaclp": "data",
    "factwacly": "data",
    "alcomx": "data",
    "ancomx": "data",
}

OUTPUTS = {
    "maut": (),
    "mfreeze": (),
    "waclp": ("plot",),
    "zaclp": ("plot",),
    "paclp": ("plot",),
    "delimx": (),
    "drlimx": (),
    "yyd": (),
    "yy": (),
    "zzd": (),
    "zz": (),
    "delecx": (),
    "delrcx": (),
    "alcomx_actual": ("plot",),
    "ancomx_actual": ("plot",),
    "GAINFP": ("plot",),
    "gainl": (),
    "gkp": (),
    "gkphi": (),
    "isetc2": (),
    "wacly": ("plot",),
    "zacly": (),
    "pacly": (),
    "GAINFY": (),
    "factwaclp": (),
    "factwacly": (),
    "alcomx": ("plot",),
    "ancomx": ("plot",),
}

INT_FIELDS = ("maut", "mfreeze")
VEC_FIELDS = ("GAINFP", "GAINFY")
NOT_DEFINED = (
    "dalimx",
    "mroll",
    "philimx",
    "phicomx",
    "delacx",
    "alimitx",
    "pdynmc",
    "FSPCB",
    "qqcx",
    "rrcx",
    "dvbec",
    "dvbe",
    "mprop",
    "gnmax",
    "gymax",
    "dla",
    "dma",
    "dmq",
    "dmde",
    "dyb",
    "dnb",
    "dnr",
    "dndr",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


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
    mprop=MPROP,
    gnmax=GNMAX,
    gymax=GYMAX,
    pdynmc=PDYNMC,
    dla=DLA,
    dma=DMA,
    dmq=DMQ,
    dmde=DMDE,
    dyb=DYB,
    dnb=DNB,
    dnr=DNR,
    dndr=DNDR,
    dvbec=DVBEC,
    dvbe=DVBE,
    qqcx=QQCX,
    rrcx=RRCX,
    fspcb=FSPCB,
):
    store.define(Field("mprop", mprop, "int", "data", "propulsion"))
    store.define(Field("gnmax", gnmax, "real", "out", "aerodynamics"))
    store.define(Field("gymax", gymax, "real", "out", "aerodynamics"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("dla", dla, "real", "out", "aerodynamics"))
    store.define(Field("dma", dma, "real", "out", "aerodynamics"))
    store.define(Field("dmq", dmq, "real", "out", "aerodynamics"))
    store.define(Field("dmde", dmde, "real", "out", "aerodynamics"))
    store.define(Field("dyb", dyb, "real", "out", "aerodynamics"))
    store.define(Field("dnb", dnb, "real", "out", "aerodynamics"))
    store.define(Field("dnr", dnr, "real", "out", "aerodynamics"))
    store.define(Field("dndr", dndr, "real", "out", "aerodynamics"))
    store.define(Field("dvbec", dvbec, "real", "out", "ins"))
    store.define(Field("dvbe", dvbe, "real", "out", "newton"))
    store.define(Field("qqcx", qqcx, "real", "out", "ins"))
    store.define(Field("rrcx", rrcx, "real", "out", "ins"))
    store.define(Field("FSPCB", fspcb, "vec", "out", "ins"))


def _ready(
    *,
    maut=MAUT,
    delimx=DELIMX,
    drlimx=DRLIMX,
    zaclp=ZACLP,
    zacly=ZACLY,
    factwaclp=FACTWACLP,
    factwacly=FACTWACLY,
    ancomx=ANCOMX,
    alcomx=ALCOMX,
    mprop=MPROP,
    gnmax=GNMAX,
    gymax=GYMAX,
    pdynmc=PDYNMC,
    dla=DLA,
    dma=DMA,
    dmq=DMQ,
    dmde=DMDE,
    dyb=DYB,
    dnb=DNB,
    dnr=DNR,
    dndr=DNDR,
    dvbec=DVBEC,
    dvbe=DVBE,
    qqcx=QQCX,
    rrcx=RRCX,
    fspcb=FSPCB,
    **states,
):
    vehicle = _Vehicle()
    control = Rocket6Control()
    control.define(vehicle)
    _plant_externals(
        vehicle.store,
        mprop=mprop,
        gnmax=gnmax,
        gymax=gymax,
        pdynmc=pdynmc,
        dla=dla,
        dma=dma,
        dmq=dmq,
        dmde=dmde,
        dyb=dyb,
        dnb=dnb,
        dnr=dnr,
        dndr=dndr,
        dvbec=dvbec,
        dvbe=dvbe,
        qqcx=qqcx,
        rrcx=rrcx,
        fspcb=fspcb,
    )
    store = vehicle.store
    store.set("maut", maut)
    store.set("delimx", delimx)
    store.set("drlimx", drlimx)
    store.set("zaclp", zaclp)
    store.set("zacly", zacly)
    store.set("factwaclp", factwaclp)
    store.set("factwacly", factwacly)
    store.set("ancomx", ancomx)
    store.set("alcomx", alcomx)
    for name, value in states.items():
        store.set(name, value)
    control.initialize(vehicle, _ctx())
    return vehicle, control


def test_name_is_control():
    assert Rocket6Control().name == "control"


def test_define_registers_cpp_fields_not_externals():
    vehicle = _Vehicle()
    Rocket6Control().define(vehicle)
    store = vehicle.store
    for name in DEFINED:
        assert name in store.names(), name
        field = store.field(name)
        assert field.module == "control"
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        if name in INT_FIELDS:
            assert field.type == "int"
            assert store.get(name) == 0
        elif name in VEC_FIELDS:
            assert field.type == "vec"
            np.testing.assert_array_equal(store.get(name), np.zeros(3))
        else:
            assert field.type == "real"
            assert store.get(name) == 0.0
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_initialize_is_noop():
    vehicle = _Vehicle()
    control = Rocket6Control()
    control.define(vehicle)
    store = vehicle.store
    store.set("maut", 53)
    store.set("ancomx", ANCOMX)
    control.initialize(vehicle, _ctx())
    assert store.get("maut") == 53
    assert store.get("ancomx") == ANCOMX
    assert store.get("delecx") == 0.0
    assert store.get("delrcx") == 0.0


def test_frozen_pdynmc_dla_dmde_fspcb_matches_cpp_delecx():
    # Break: wrong online waclp/paclp, pitch FSPCB index, or dvbec vs dvbe.
    vehicle, control = _ready()
    control.execute(vehicle, _ctx(DT))
    store = vehicle.store
    assert _approx(store.get("delecx"), CPP_DELECX)
    assert _approx(store.get("delrcx"), CPP_DELRCX)
    assert _approx(store.get("waclp"), CPP_WACLP)
    assert _approx(store.get("paclp"), CPP_PACLP)
    np.testing.assert_allclose(store.get("GAINFP"), CPP_GAINFP, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("GAINFY"), CPP_GAINFY, rtol=RTOL, atol=ATOL)
    assert _approx(store.get("zzd"), CPP_ZZD)
    assert _approx(store.get("zz"), CPP_ZZ)
    assert _approx(store.get("yyd"), CPP_YYD)
    assert _approx(store.get("yy"), CPP_YY)


def test_maut_53_no_error():
    vehicle, control = _ready(maut=53)
    control.execute(vehicle, _ctx(DT))
    assert np.isfinite(vehicle.store.get("delecx"))
    assert np.isfinite(vehicle.store.get("delrcx"))
    assert abs(vehicle.store.get("delecx")) <= DELIMX
    assert abs(vehicle.store.get("delrcx")) <= DRLIMX


def test_maut_24_raises():
    # Break: Hyper6 climb maut=24 accepted (yaw-rate + gamma).
    vehicle, control = _ready(maut=24)
    with pytest.raises(ValueError):
        control.execute(vehicle, _ctx())


def test_other_maut_raises():
    for maut in (1, 3, 5, 50, -1):
        vehicle, control = _ready(maut=maut)
        with pytest.raises(ValueError):
            control.execute(vehicle, _ctx())


def test_maut_0_finite_limited_commands():
    # Break: Hyper6-style early return leaves stale commands, or skip write.
    vehicle, control = _ready(maut=0, ancomx=ANCOMX, alcomx=ALCOMX)
    control.execute(vehicle, _ctx(DT))
    store = vehicle.store
    assert np.isfinite(store.get("delecx"))
    assert np.isfinite(store.get("delrcx"))
    assert abs(store.get("delecx")) <= DELIMX
    assert abs(store.get("delrcx")) <= DRLIMX
    assert store.get("delecx") == 0.0
    assert store.get("delrcx") == 0.0
    assert store.get("ancomx_actual") == ANCOMX
    assert store.get("alcomx_actual") == ALCOMX
    assert store.get("zz") == 0.0
    assert store.get("yy") == 0.0


def test_mprop_0_skips_accel_calls():
    # Break: C++ if(mprop) ignored; accel still writes delecx.
    vehicle, control = _ready(maut=53, mprop=0)
    control.execute(vehicle, _ctx(DT))
    store = vehicle.store
    assert store.get("delecx") == 0.0
    assert store.get("delrcx") == 0.0
    assert store.get("zz") == 0.0
    assert store.get("yy") == 0.0
    np.testing.assert_array_equal(store.get("GAINFP"), np.zeros(3))
    assert store.get("ancomx_actual") == ANCOMX
    assert store.get("alcomx_actual") == ALCOMX


def test_yaw_uses_newton_dvbe_not_ins_dvbec():
    # Break: yaw copies pitch and reads dvbec. Frozen dvbe=200 vs dvbec=170.
    vehicle, control = _ready()
    control.execute(vehicle, _ctx(DT))
    got = vehicle.store.get("delrcx")
    assert _approx(got, CPP_DELRCX)
    assert got != pytest.approx(CPP_DELRCX_IF_DVBEC, rel=RTOL, abs=ATOL)


def test_pitch_uses_fspcb_index_2():
    # Break: pitch reads FSPCB[1] (yaw component).
    vehicle, control = _ready()
    control.execute(vehicle, _ctx(DT))
    assert _approx(vehicle.store.get("delecx"), CPP_DELECX)
    assert vehicle.store.get("delecx") != pytest.approx(0.5701851505101283, rel=RTOL, abs=ATOL)


def test_command_limiter_cadac_sign():
    vehicle, control = _ready(maut=53, delimx=1.0, drlimx=0.1)
    control.execute(vehicle, _ctx(DT))
    store = vehicle.store
    assert _approx(store.get("delecx"), -1.0)
    assert _approx(store.get("delrcx"), -0.1)


def test_maut_0_negative_limiter_uses_cadac_sign_zero():
    # |delecx=0| > delimx when delimx < 0. CADAC sign(0)=+1 → delimx*1.
    # np.sign(0)=0 would leave 0.
    vehicle, control = _ready(maut=0, delimx=-10.0, drlimx=-10.0)
    control.execute(vehicle, _ctx(DT))
    store = vehicle.store
    assert _approx(store.get("delecx"), -10.0)
    assert _approx(store.get("delrcx"), -10.0)
    assert np.sign(0.0) == 0.0


def test_gnmax_gymax_clamp_commands():
    vehicle, control = _ready(maut=53, gnmax=0.05, gymax=0.05)
    control.execute(vehicle, _ctx(DT))
    store = vehicle.store
    assert _approx(store.get("ancomx_actual"), -0.05)
    assert _approx(store.get("alcomx_actual"), 0.05)
    assert store.get("ancomx") == ANCOMX
    assert store.get("alcomx") == ALCOMX
    assert _approx(store.get("delecx"), CPP_DELECX_GNMAX)
    assert _approx(store.get("delrcx"), CPP_DELRCX_GYMAX)


def test_stored_slope_second_step():
    vehicle, control = _ready()
    ctx = _ctx(DT)
    control.execute(vehicle, ctx)
    assert _approx(vehicle.store.get("delecx"), CPP_DELECX)
    control.execute(vehicle, ctx)
    assert _approx(vehicle.store.get("delecx"), CPP_DELECX_STEP2)
    assert _approx(vehicle.store.get("delrcx"), CPP_DELRCX_STEP2)
    assert vehicle.store.get("delecx") != pytest.approx(CPP_DELECX, rel=RTOL, abs=ATOL)


def test_ports_accel_helpers():
    control = Rocket6Control()
    assert callable(getattr(control, "control_normal_accel"))
    assert callable(getattr(control, "control_yaw_accel"))


def test_terminate_exists_and_is_pass():
    vehicle, control = _ready(maut=0)
    assert control.terminate(vehicle, _ctx()) is None

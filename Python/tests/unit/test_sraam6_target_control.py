import math

import numpy as np

from cadac.constants import DEG, EPS, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.sraam6.target import Sraam6TargetControl

RTOL = 1e-12
ATOL = 1e-14

GRAV = 9.8
GTURN = 1.0
PHILIMX = 120.0
ALPLIMX = 40.0
CLALPHA = 0.0523
WINGLOADING = 3247.0
# Above clip for g-turn ancomx=sqrt(2) with default limiter.
PDYNMC = 50000.0

DEFINED = (
    "phiav",
    "phiavd",
    "tphi",
    "philimx",
    "phiavx",
    "phiavcx",
    "anx",
    "anxd",
    "tanx",
    "alplimx",
    "ancomx",
    "clalpha",
    "wingloading",
    "phiavout",
)
STATE = ("phiav", "phiavd", "anx", "anxd")
DATA = ("tphi", "philimx", "tanx", "alplimx", "clalpha", "wingloading")
OUT = ("phiavx", "phiavout")
DIAG = ("phiavcx", "ancomx")
COM_OUT = ("phiavx", "anx")
EXTERNALS = ("grav", "pdynmc", "TVL", "tgt_option", "ACOML")


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
        vehicle_slot=1,
    )


def _cadac_sign(variable):
    if variable < 0.0:
        return -1
    return 1


def _gturn_acoml(gturn=GTURN, grav=GRAV):
    return np.array([0.0, gturn * grav, -grav])


def _plant_externals(
    store,
    *,
    grav=GRAV,
    pdynmc=PDYNMC,
    tvl=None,
    tgt_option=1,
    acoml=None,
):
    if tvl is None:
        tvl = np.eye(3)
    if acoml is None:
        acoml = _gturn_acoml(grav=grav)
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("TVL", tvl, "mat", "out", "newton"))
    store.define(Field("tgt_option", tgt_option, "int", "data", "guidance"))
    store.define(Field("ACOML", acoml, "vec", "out", "guidance"))


def _ready(
    *,
    tphi=0.0,
    tanx=0.0,
    tgt_option=1,
    grav=GRAV,
    pdynmc=PDYNMC,
    tvl=None,
    acoml=None,
    int_step=0.001,
    plant=True,
):
    vehicle = _Vehicle()
    control = Sraam6TargetControl()
    control.define(vehicle)
    if plant:
        _plant_externals(
            vehicle.store,
            grav=grav,
            pdynmc=pdynmc,
            tvl=tvl,
            tgt_option=tgt_option,
            acoml=acoml,
        )
    store = vehicle.store
    store.set("tphi", tphi)
    store.set("tanx", tanx)
    control.initialize(vehicle, _ctx(int_step))
    return vehicle, control, _ctx(int_step)


def test_name_is_control():
    assert Sraam6TargetControl().name == "control"


def test_define_registers_def_control_fields():
    vehicle = _Vehicle()
    Sraam6TargetControl().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    for name in STATE:
        assert store.field(name).type == "real"
        assert store.field(name).role == "state"
        assert store.field(name).module == "control"
        assert store.get(name) == 0.0
    for name in DATA:
        assert store.field(name).type == "real"
        assert store.field(name).role == "data"
        assert store.field(name).module == "control"
    assert store.get("tphi") == 0.0
    assert store.get("tanx") == 0.0
    np.testing.assert_allclose(store.get("philimx"), PHILIMX, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("alplimx"), ALPLIMX, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("clalpha"), CLALPHA, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        store.get("wingloading"), WINGLOADING, rtol=RTOL, atol=ATOL
    )
    for name in OUT:
        assert store.field(name).type == "real"
        assert store.field(name).role == "out"
        assert store.field(name).module == "control"
        assert store.get(name) == 0.0
    for name in DIAG:
        assert store.field(name).type == "real"
        assert store.field(name).role == "diag"
        assert store.field(name).module == "control"
        assert store.get(name) == 0.0
    for name in COM_OUT:
        assert store.field(name).outputs == ("com",)
    assert store.field("phiavout").outputs == ()
    for name in EXTERNALS:
        assert name not in store.names()


def test_gturn_identity_tvl_finite_bank_and_load():
    acoml = _gturn_acoml()
    vehicle, control, ctx = _ready(
        tphi=0.0,
        tanx=0.0,
        tgt_option=1,
        tvl=np.eye(3),
        acoml=acoml,
    )
    control.execute(vehicle, ctx)
    store = vehicle.store
    phiavx = store.get("phiavx")
    anx = store.get("anx")
    assert math.isfinite(phiavx)
    assert math.isfinite(anx)
    assert abs(phiavx) <= PHILIMX

    acoma2 = GTURN * GRAV
    acoma3 = -GRAV
    phiavc = math.atan2(acoma2, -acoma3)
    want_phiavx = phiavc * DEG
    want_ancomx = math.sqrt(acoma2 * acoma2 + acoma3 * acoma3) / GRAV
    np.testing.assert_allclose(phiavx, want_phiavx, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(anx, want_ancomx, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        store.get("phiavout"), want_phiavx * RAD, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(store.get("phiavcx"), want_phiavx, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ancomx"), want_ancomx, rtol=RTOL, atol=ATOL)


def test_bank_lags_command_when_tphi_positive():
    acoml = _gturn_acoml()
    int_step = 0.001
    tphi = 0.1
    vehicle, control, ctx = _ready(
        tphi=tphi,
        tanx=0.0,
        tgt_option=1,
        tvl=np.eye(3),
        acoml=acoml,
        int_step=int_step,
    )
    control.execute(vehicle, ctx)
    acoma2 = GTURN * GRAV
    acoma3 = -GRAV
    phiavc = math.atan2(acoma2, -acoma3)
    phiavd_new = (phiavc - 0.0) / tphi
    phiav = integrate(phiavd_new, 0.0, 0.0, int_step)
    want_phiavx = phiav * DEG
    np.testing.assert_allclose(
        vehicle.store.get("phiavx"), want_phiavx, rtol=RTOL, atol=ATOL
    )
    command = phiavc * DEG
    assert abs(vehicle.store.get("phiavx")) < abs(command)


def test_eps_zero_acoma_gives_zero_bank():
    acoml = np.array([1.0, 0.5 * EPS, -0.5 * EPS])
    vehicle, control, ctx = _ready(
        tphi=0.0, tanx=0.0, tgt_option=1, tvl=np.eye(3), acoml=acoml
    )
    control.execute(vehicle, ctx)
    np.testing.assert_allclose(vehicle.store.get("phiavx"), 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("phiavcx"), 0.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("phiavout"), 0.0, rtol=RTOL, atol=ATOL)


def test_philimx_clips_bank_with_cadac_sign():
    acoml = np.array([0.0, 1.0, 1.0])
    vehicle, control, ctx = _ready(
        tphi=0.0, tanx=0.0, tgt_option=0, tvl=np.eye(3), acoml=acoml
    )
    control.execute(vehicle, ctx)
    command = math.atan2(1.0, -1.0) * DEG
    assert abs(command) > PHILIMX
    np.testing.assert_allclose(
        vehicle.store.get("phiavx"),
        PHILIMX * _cadac_sign(command),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        vehicle.store.get("phiav"), command * RAD, rtol=RTOL, atol=ATOL
    )

    acoml_neg = np.array([0.0, -1.0, 1.0])
    vehicle_n, control_n, ctx_n = _ready(
        tphi=0.0, tanx=0.0, tgt_option=0, tvl=np.eye(3), acoml=acoml_neg
    )
    control_n.execute(vehicle_n, ctx_n)
    command_n = math.atan2(-1.0, -1.0) * DEG
    np.testing.assert_allclose(
        vehicle_n.store.get("phiavx"),
        PHILIMX * _cadac_sign(command_n),
        rtol=RTOL,
        atol=ATOL,
    )


def test_tgt_option_positive_applies_alpha_limiter():
    pdynmc = 1000.0
    acoml = np.array([0.0, 0.0, -10.0 * GRAV])
    vehicle, control, ctx = _ready(
        tphi=0.0,
        tanx=0.0,
        tgt_option=1,
        pdynmc=pdynmc,
        tvl=np.eye(3),
        acoml=acoml,
    )
    control.execute(vehicle, ctx)
    anlimx = pdynmc * CLALPHA * ALPLIMX / (WINGLOADING * GRAV)
    ancomx = 10.0
    assert ancomx > anlimx
    np.testing.assert_allclose(
        vehicle.store.get("ancomx"), ancomx, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("anx"),
        anlimx * _cadac_sign(ancomx),
        rtol=RTOL,
        atol=ATOL,
    )


def test_tgt_option_zero_skips_alpha_limiter():
    pdynmc = 1000.0
    acoml = np.array([0.0, 0.0, -10.0 * GRAV])
    vehicle, control, ctx = _ready(
        tphi=0.0,
        tanx=0.0,
        tgt_option=0,
        pdynmc=pdynmc,
        tvl=np.eye(3),
        acoml=acoml,
    )
    control.execute(vehicle, ctx)
    np.testing.assert_allclose(vehicle.store.get("anx"), 10.0, rtol=RTOL, atol=ATOL)


def test_tanx_lags_load_factor():
    tanx = 0.1
    int_step = 0.001
    acoml = _gturn_acoml()
    vehicle, control, ctx = _ready(
        tphi=0.0,
        tanx=tanx,
        tgt_option=0,
        tvl=np.eye(3),
        acoml=acoml,
        int_step=int_step,
    )
    control.execute(vehicle, ctx)
    acoma2 = GTURN * GRAV
    acoma3 = -GRAV
    ancomx = math.sqrt(acoma2 * acoma2 + acoma3 * acoma3) / GRAV
    anxd_new = (ancomx - 0.0) / tanx
    want_anx = integrate(anxd_new, 0.0, 0.0, int_step)
    np.testing.assert_allclose(
        vehicle.store.get("anx"), want_anx, rtol=RTOL, atol=ATOL
    )
    assert abs(vehicle.store.get("anx")) < abs(ancomx)

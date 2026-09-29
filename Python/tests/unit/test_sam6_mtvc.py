from math import cos, exp, sin

import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.sam6.forces import Sam6Forces
from cadac.vehicles.flat6.sam6.tvc import Sam6Tvc

RTOL = 1e-12
ATOL = 1e-14

DT = 0.001
GTVC0 = 0.5
PARM = 2.5
XCG = 1.0
TVCLIMX = 10.0
DTVCLIMX = 200.0
WNTVC = 200.0
ZETTVC = 0.7
PDYNMC_GTVC36 = 1000.0
THRUST = 1000.0
DQCX = 1.0
DRCX = 0.5
PDYNMC = 156.0
REFA = 0.0491
REFL = 0.25
CA = 0.4


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


def _plant_tvc_externals(
    store,
    *,
    mprop=1,
    maut=0,
    thrust=THRUST,
    xcg=XCG,
    dqcx=DQCX,
    drcx=DRCX,
    dqcx_rcs=0.0,
    drcx_rcs=0.0,
    pdynmc=PDYNMC,
):
    store.define(Field("mprop", mprop, "int", "out", "propulsion"))
    store.define(Field("maut", maut, "int", "data", "control"))
    store.define(Field("thrust", thrust, "real", "out", "propulsion"))
    store.define(Field("xcg", xcg, "real", "out", "propulsion"))
    store.define(Field("dqcx", dqcx, "real", "out", "control"))
    store.define(Field("drcx", drcx, "real", "out", "control"))
    store.define(Field("dqcx_rcs", dqcx_rcs, "real", "out", "control"))
    store.define(Field("drcx_rcs", drcx_rcs, "real", "out", "control"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))


def _ready_tvc(
    *,
    mtvc=2,
    gtvc0=GTVC0,
    parm=PARM,
    tvclimx=TVCLIMX,
    dtvclimx=DTVCLIMX,
    wntvc=WNTVC,
    zettvc=ZETTVC,
    pdynmc_gtvc36=PDYNMC_GTVC36,
    mprop=1,
    maut=0,
    thrust=THRUST,
    xcg=XCG,
    dqcx=DQCX,
    drcx=DRCX,
    dqcx_rcs=0.0,
    drcx_rcs=0.0,
    pdynmc=PDYNMC,
    plant=True,
    **states,
):
    vehicle = _Vehicle()
    tvc = Sam6Tvc()
    tvc.define(vehicle)
    if plant:
        _plant_tvc_externals(
            vehicle.store,
            mprop=mprop,
            maut=maut,
            thrust=thrust,
            xcg=xcg,
            dqcx=dqcx,
            drcx=drcx,
            dqcx_rcs=dqcx_rcs,
            drcx_rcs=drcx_rcs,
            pdynmc=pdynmc,
        )
    store = vehicle.store
    store.set("mtvc", mtvc)
    store.set("gtvc0", gtvc0)
    store.set("parm", parm)
    store.set("tvclimx", tvclimx)
    store.set("dtvclimx", dtvclimx)
    store.set("wntvc", wntvc)
    store.set("zettvc", zettvc)
    store.set("pdynmc_gtvc36", pdynmc_gtvc36)
    for name, value in states.items():
        store.set(name, value)
    tvc.initialize(vehicle, _ctx())
    return vehicle, tvc


def _axis_scnd(command, pos_d, pos, rate_d, rate, tvclimx, dtvclimx, wntvc, zettvc, dt):
    if abs(pos) > tvclimx * RAD:
        pos = tvclimx * RAD * _sign(pos)
        if pos * rate > 0.0:
            rate = 0.0
    iflag = 0
    if abs(rate) > dtvclimx * RAD:
        iflag = 1
        rate = dtvclimx * RAD * _sign(rate)
    pos_d_new = rate
    pos = integrate(pos_d_new, pos_d, pos, dt)
    pos_d = pos_d_new
    err = command - pos
    rate_d_new = wntvc * wntvc * err - 2.0 * zettvc * wntvc * pos_d
    rate = integrate(rate_d_new, rate_d, rate, dt)
    rate_d = rate_d_new
    if iflag and rate * rate_d > 0.0:
        rate_d = 0.0
    return pos, pos_d, rate, rate_d


def _expected_tvc(store, dt):
    mtvc = store.get("mtvc")
    mprop = store.get("mprop") if "mprop" in store else 0
    if not (mtvc > 0 and mprop > 0):
        return {
            "FPB": np.zeros(3),
            "FMPB": np.zeros(3),
            "gtvc": 0.0,
            "etax": 0.0,
            "zetx": 0.0,
            "etacx": 0.0,
            "zetcx": 0.0,
            "etasd": store.get("etasd"),
            "zetad": store.get("zetad"),
            "etas": store.get("etas"),
            "zeta": store.get("zeta"),
            "detasd": store.get("detasd"),
            "dzetad": store.get("dzetad"),
            "detas": store.get("detas"),
            "dzeta": store.get("dzeta"),
        }
    if mtvc not in (1, 2, 3):
        raise ValueError(f"unknown mtvc {mtvc}")
    gtvc = 0.0
    if mtvc == 2:
        gtvc = store.get("gtvc0")
    if mtvc == 3:
        gtvc = store.get("gtvc0") * exp(-store.get("pdynmc") / store.get("pdynmc_gtvc36"))
    dqcx = store.get("dqcx")
    drcx = store.get("drcx")
    if store.get("maut") == 4:
        dqcx = store.get("dqcx_rcs")
        drcx = store.get("drcx_rcs")
    etac = gtvc * dqcx * RAD
    zetc = gtvc * drcx * RAD
    if mtvc == 1:
        eta = etac
        zet = zetc
        etasd = store.get("etasd")
        zetad = store.get("zetad")
        etas = store.get("etas")
        zeta = store.get("zeta")
        detasd = store.get("detasd")
        dzetad = store.get("dzetad")
        detas = store.get("detas")
        dzeta = store.get("dzeta")
    else:
        etas, etasd, detas, detasd = _axis_scnd(
            etac,
            store.get("etasd"),
            store.get("etas"),
            store.get("detasd"),
            store.get("detas"),
            store.get("tvclimx"),
            store.get("dtvclimx"),
            store.get("wntvc"),
            store.get("zettvc"),
            dt,
        )
        zeta, zetad, dzeta, dzetad = _axis_scnd(
            zetc,
            store.get("zetad"),
            store.get("zeta"),
            store.get("dzetad"),
            store.get("dzeta"),
            store.get("tvclimx"),
            store.get("dtvclimx"),
            store.get("wntvc"),
            store.get("zettvc"),
            dt,
        )
        eta = etas
        zet = zeta
    thrust = store.get("thrust")
    fpb0 = cos(eta) * cos(zet) * thrust
    fpb1 = cos(eta) * sin(zet) * thrust
    fpb2 = -sin(eta) * thrust
    arm = store.get("parm") - store.get("xcg")
    return {
        "FPB": np.array([fpb0, fpb1, fpb2], dtype=float),
        "FMPB": np.array([0.0, arm * fpb2, -arm * fpb1], dtype=float),
        "gtvc": gtvc,
        "etax": eta * DEG,
        "zetx": zet * DEG,
        "etacx": etac * DEG,
        "zetcx": zetc * DEG,
        "etasd": etasd,
        "zetad": zetad,
        "etas": etas,
        "zeta": zeta,
        "detasd": detasd,
        "dzetad": dzetad,
        "detas": detas,
        "dzeta": dzeta,
    }


def _assert_tvc_step(store, want):
    for name in (
        "gtvc",
        "etax",
        "zetx",
        "etacx",
        "zetcx",
        "etasd",
        "zetad",
        "etas",
        "zeta",
        "detasd",
        "dzetad",
        "detas",
        "dzeta",
    ):
        assert np.isfinite(store.get(name)), name
        assert _approx(store.get(name), want[name]), name
    np.testing.assert_allclose(store.get("FPB"), want["FPB"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FMPB"), want["FMPB"], rtol=RTOL, atol=ATOL)


def test_sam6_mtvc_2_first_step_matches_cadac():
    vehicle, tvc = _ready_tvc(mtvc=2, dqcx=DQCX, drcx=DRCX)
    want = _expected_tvc(vehicle.store, DT)
    tvc.execute(vehicle, _ctx(DT))
    _assert_tvc_step(vehicle.store, want)
    assert want["gtvc"] == GTVC0
    assert abs(vehicle.store.get("etax")) <= TVCLIMX
    assert np.isfinite(vehicle.store.get("FPB")[0])


def test_sam6_mtvc_2_fpb_fmpb_from_second_order_eta():
    vehicle, tvc = _ready_tvc(mtvc=2, dqcx=DQCX, drcx=0.5)
    want = _expected_tvc(vehicle.store, DT)
    tvc.execute(vehicle, _ctx(DT))
    store = vehicle.store
    eta = store.get("etas")
    zet = store.get("zeta")
    fpb = store.get("FPB")
    fmpb = store.get("FMPB")
    assert _approx(fpb[0], cos(eta) * cos(zet) * THRUST)
    assert _approx(fpb[1], cos(eta) * sin(zet) * THRUST)
    assert _approx(fpb[2], -sin(eta) * THRUST)
    arm = PARM - XCG
    assert fmpb[0] == 0.0
    assert _approx(fmpb[1], arm * fpb[2])
    assert _approx(fmpb[2], -arm * fpb[1])
    assert _approx(store.get("etacx"), GTVC0 * DQCX)
    assert _approx(store.get("zetcx"), GTVC0 * 0.5)
    np.testing.assert_allclose(fpb, want["FPB"], rtol=RTOL, atol=ATOL)


def test_sam6_mtvc_1_no_dynamics_gtvc_stays_zero():
    # C++ only assigns gtvc for mtvc==2/3; mtvc==1 keeps local gtvc=0 → eta=zet=0.
    vehicle, tvc = _ready_tvc(mtvc=1, dqcx=DQCX, drcx=DRCX, thrust=THRUST)
    want = _expected_tvc(vehicle.store, DT)
    tvc.execute(vehicle, _ctx(DT))
    _assert_tvc_step(vehicle.store, want)
    assert want["gtvc"] == 0.0
    np.testing.assert_allclose(
        vehicle.store.get("FPB"),
        np.array([THRUST, 0.0, 0.0]),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        vehicle.store.get("FMPB"), np.zeros(3), rtol=RTOL, atol=ATOL
    )
    assert vehicle.store.get("etas") == 0.0
    assert vehicle.store.get("detas") == 0.0


def test_sam6_mtvc_3_variable_gain_exp_pdynmc():
    vehicle, tvc = _ready_tvc(
        mtvc=3,
        dqcx=DQCX,
        drcx=0.0,
        pdynmc=PDYNMC,
        pdynmc_gtvc36=PDYNMC_GTVC36,
        gtvc0=GTVC0,
    )
    want = _expected_tvc(vehicle.store, DT)
    gtvc_want = GTVC0 * exp(-PDYNMC / PDYNMC_GTVC36)
    assert want["gtvc"] == pytest.approx(gtvc_want, rel=RTOL, abs=ATOL)
    assert want["gtvc"] != pytest.approx(GTVC0, rel=1e-6)
    tvc.execute(vehicle, _ctx(DT))
    _assert_tvc_step(vehicle.store, want)
    assert _approx(vehicle.store.get("etacx"), gtvc_want * DQCX)


def test_sam6_mtvc_maut4_uses_rcs_commands():
    vehicle, tvc = _ready_tvc(
        mtvc=2,
        maut=4,
        dqcx=99.0,
        drcx=99.0,
        dqcx_rcs=2.0,
        drcx_rcs=-1.0,
    )
    want = _expected_tvc(vehicle.store, DT)
    tvc.execute(vehicle, _ctx(DT))
    _assert_tvc_step(vehicle.store, want)
    assert _approx(vehicle.store.get("etacx"), GTVC0 * 2.0)
    assert _approx(vehicle.store.get("zetcx"), GTVC0 * -1.0)


def test_sam6_mtvc_mprop_zero_leaves_fpb_zero():
    vehicle, tvc = _ready_tvc(mtvc=2, mprop=0, thrust=THRUST, dqcx=DQCX)
    tvc.execute(vehicle, _ctx(DT))
    np.testing.assert_allclose(vehicle.store.get("FPB"), np.zeros(3), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("FMPB"), np.zeros(3), rtol=RTOL, atol=ATOL)
    assert vehicle.store.get("gtvc") == 0.0
    assert vehicle.store.get("etas") == 0.0


def test_sam6_mtvc_zero_rewrites_fpb_zero():
    vehicle, tvc = _ready_tvc(mtvc=0, thrust=THRUST, dqcx=DQCX)
    vehicle.store.set("FPB", (9.0, 8.0, 7.0))
    vehicle.store.set("FMPB", (6.0, 5.0, 4.0))
    tvc.execute(vehicle, _ctx(DT))
    np.testing.assert_allclose(vehicle.store.get("FPB"), np.zeros(3), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vehicle.store.get("FMPB"), np.zeros(3), rtol=RTOL, atol=ATOL)


def test_sam6_mtvc_unknown_raises():
    for mtvc in (-1, 4):
        vehicle, tvc = _ready_tvc(mtvc=mtvc, mprop=1)
        with pytest.raises(ValueError):
            tvc.execute(vehicle, _ctx())


def test_sam6_mtvc_2_second_step_advances_state():
    vehicle, tvc = _ready_tvc(mtvc=2, dqcx=DQCX, drcx=0.5)
    ctx = _ctx(DT)
    want1 = _expected_tvc(vehicle.store, DT)
    tvc.execute(vehicle, ctx)
    _assert_tvc_step(vehicle.store, want1)
    want2 = _expected_tvc(vehicle.store, DT)
    tvc.execute(vehicle, ctx)
    _assert_tvc_step(vehicle.store, want2)
    assert vehicle.store.get("etas") != want1["etas"]


def test_sam6_mtvc_position_limit_cadac_sign():
    vehicle, tvc = _ready_tvc(mtvc=2, dqcx=DQCX, etas=0.0, tvclimx=-5.0, detas=1.0)
    want = _expected_tvc(vehicle.store, DT)
    cadac_clamped = -5.0 * RAD * _sign(0.0)
    numpy_clamped = -5.0 * RAD * float(np.sign(0.0))
    assert cadac_clamped != numpy_clamped
    tvc.execute(vehicle, _ctx(DT))
    _assert_tvc_step(vehicle.store, want)


def _plant_forces(
    store,
    *,
    mtvc=0,
    pdynmc=PDYNMC,
    refa=REFA,
    refl=REFL,
    ca=CA,
    cy=0.0,
    cn=0.0,
    cll=0.0,
    clm=0.0,
    cln=0.0,
    thrust=THRUST,
    fpb=(0.0, 0.0, 0.0),
    fmpb=(0.0, 0.0, 0.0),
    farcs=None,
    fmrcs=None,
):
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("refa", refa, "real", "init", "aerodynamics"))
    store.define(Field("refl", refl, "real", "init", "aerodynamics"))
    store.define(Field("ca", ca, "real", "out", "aerodynamics"))
    store.define(Field("cy", cy, "real", "out", "aerodynamics"))
    store.define(Field("cn", cn, "real", "out", "aerodynamics"))
    store.define(Field("cll", cll, "real", "out", "aerodynamics"))
    store.define(Field("clm", clm, "real", "out", "aerodynamics"))
    store.define(Field("cln", cln, "real", "out", "aerodynamics"))
    store.define(Field("thrust", thrust, "real", "out", "propulsion"))
    store.define(Field("mtvc", mtvc, "int", "data", "tvc"))
    store.define(Field("FPB", fpb, "vec", "out", "tvc"))
    store.define(Field("FMPB", fmpb, "vec", "out", "tvc"))
    if farcs is not None:
        store.define(Field("FARCS", farcs, "vec", "out", "rcs"))
    if fmrcs is not None:
        store.define(Field("FMRCS", fmrcs, "vec", "out", "rcs"))


def _ready_forces(**kwargs):
    vehicle = _Vehicle()
    forces = Sam6Forces()
    forces.define(vehicle)
    forces.initialize(vehicle, _ctx())
    _plant_forces(vehicle.store, **kwargs)
    return vehicle, forces


@pytest.mark.parametrize("mtvc", [1, 2, 3])
def test_sam6_forces_mtvc_adds_fpb_not_raw_thrust(mtvc):
    fpb = np.array([111.0, 22.0, -33.0])
    fmpb = np.array([4.0, -5.0, 6.0])
    vehicle, forces = _ready_forces(mtvc=mtvc, fpb=fpb, fmpb=fmpb, thrust=THRUST)
    forces.execute(vehicle, _ctx())
    store = vehicle.store
    want_fapb = np.array(
        [
            -PDYNMC * REFA * CA + fpb[0],
            PDYNMC * REFA * 0.0 + fpb[1],
            -PDYNMC * REFA * 0.0 + fpb[2],
        ],
        dtype=float,
    )
    want_fmb = np.array(
        [
            PDYNMC * REFA * REFL * 0.0 + fmpb[0],
            PDYNMC * REFA * REFL * 0.0 + fmpb[1],
            PDYNMC * REFA * REFL * 0.0 + fmpb[2],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(store.get("FAPB"), want_fapb, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("FMB"), want_fmb, rtol=RTOL, atol=ATOL)
    # raw thrust must not be added when TVC owns propulsion force
    assert store.get("FAPB")[0] != pytest.approx(
        -PDYNMC * REFA * CA + THRUST, rel=RTOL, abs=ATOL
    )


def test_sam6_forces_mtvc_zero_still_adds_thrust_ignores_fpb():
    vehicle, forces = _ready_forces(
        mtvc=0,
        fpb=(50.0, 40.0, 30.0),
        fmpb=(7.0, 8.0, 9.0),
        thrust=THRUST,
    )
    forces.execute(vehicle, _ctx())
    want_fapb0 = -PDYNMC * REFA * CA + THRUST
    np.testing.assert_allclose(
        vehicle.store.get("FAPB")[0], want_fapb0, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        vehicle.store.get("FMB"), np.zeros(3), rtol=RTOL, atol=ATOL
    )

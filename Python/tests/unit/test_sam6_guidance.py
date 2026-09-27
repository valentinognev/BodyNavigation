from math import atan2, cos, exp, fabs, sin, sqrt, tan
from pathlib import Path

import numpy as np
import pytest

from cadac.constants import AGRAV, DEG, RAD
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import mat2tr, polar_from_cart
from cadac.vehicles.flat6.sam6.guidance import Sam6Guidance

RTOL = 1e-12
ATOL = 1e-14
SMALL = 1e-7
DT = 0.001
GMAX = 40.0
ZEROS3 = (0.0, 0.0, 0.0)
ZEROS33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

# RF #1 line / term-7 numbers; IR #1 term-6 gnav/tgnav.
LINE_GAIN = 1.0
NL_GAIN_FACT = 0.5
DECREMENT = 7000.0
THTFLX = 0.0
GNAV = 3.1
GNAV_IR = 3.5
TGNAV = 1.0
GRAV_BIAS = 1.0
SIEL_NORTH = np.array([1000.0, 0.0, 0.0], dtype=float)
VBELC = np.array([16.0, 5.0, -1.0], dtype=float)
SBELC = np.array([0.0, 0.0, 0.0], dtype=float)
GRAV = AGRAV

DEFINED = (
    "mguide",
    "ancomx",
    "alcomx",
    "gnav_comp",
    "grav_bias",
    "apnyx",
    "apnzx",
    "adely",
    "adelz",
    "allx",
    "annx",
    "epchta",
    "gnav",
    "gnd",
    "gn",
    "tgnav",
    "dcvel",
    "line_gain",
    "nl_gain_fact",
    "decrement",
    "psiflx",
    "thtflx",
    "ip_sltrange",
    "nl_gain",
    "VBEO",
    "VBEF",
    "SIBLC",
    "WOELC",
    "tgoc",
    "dtbc",
    "dvtbc",
    "psiobcx",
    "thtobcx",
    "UTBLC",
)
ROLES = {
    "mguide": "data",
    "ancomx": "out",
    "alcomx": "out",
    "gnav_comp": "diag",
    "grav_bias": "data",
    "apnyx": "diag",
    "apnzx": "diag",
    "adely": "diag",
    "adelz": "diag",
    "allx": "diag",
    "annx": "diag",
    "epchta": "save",
    "gnav": "data",
    "gnd": "state",
    "gn": "state",
    "tgnav": "data",
    "dcvel": "diag",
    "line_gain": "data",
    "nl_gain_fact": "data",
    "decrement": "data",
    "psiflx": "diag",
    "thtflx": "data",
    "ip_sltrange": "diag",
    "nl_gain": "diag",
    "VBEO": "diag",
    "VBEF": "diag",
    "SIBLC": "out",
    "WOELC": "out",
    "tgoc": "diag",
    "dtbc": "diag",
    "dvtbc": "diag",
    "psiobcx": "diag",
    "thtobcx": "diag",
    "UTBLC": "out",
}
OUTPUTS = {
    "mguide": (),
    "ancomx": ("scrn", "plot"),
    "alcomx": ("scrn", "plot"),
    "gnav_comp": ("plot",),
    "grav_bias": (),
    "apnyx": (),
    "apnzx": (),
    "adely": (),
    "adelz": (),
    "allx": ("plot",),
    "annx": ("plot",),
    "epchta": (),
    "gnav": (),
    "gnd": (),
    "gn": ("plot",),
    "tgnav": (),
    "dcvel": ("plot",),
    "line_gain": (),
    "nl_gain_fact": (),
    "decrement": (),
    "psiflx": ("plot",),
    "thtflx": (),
    "ip_sltrange": ("plot",),
    "nl_gain": (),
    "VBEO": (),
    "VBEF": (),
    "SIBLC": (),
    "WOELC": (),
    "tgoc": (),
    "dtbc": (),
    "dvtbc": ("plot",),
    "psiobcx": (),
    "thtobcx": (),
    "UTBLC": (),
}
INT_FIELDS = ("mguide",)
VEC_FIELDS = ("VBEO", "VBEF", "SIBLC", "WOELC", "UTBLC")
NOT_DEFINED = (
    "gmax",
    "TBLC",
    "VBELC",
    "SBELC",
    "thtvlcx",
    "psivlcx",
    "FSPCB",
    "grav",
    "SBEL",
    "sbel1",
    "sbel2",
    "sbel3",
    "STEL",
    "VTEL",
    "thtpb",
    "psipb",
    "sigdy",
    "sigdz",
    "ddab",
    "psisb",
    "thtsb",
    "lamdqb",
    "lamdrb",
    "launch_time",
    "psiblx",
    "ancomx_test",
    "time",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ctx(int_step=DT, combus=None, vehicle_slot=0):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=vehicle_slot,
    )


def _circular_limit(allx, annx, gmax):
    aa = sqrt(allx * allx + annx * annx)
    if aa > gmax:
        aa = gmax
    if fabs(annx) < SMALL and fabs(allx) < SMALL:
        phi = 0.0
    else:
        phi = atan2(annx, allx)
    return aa * cos(phi), aa * sin(phi)


def _cpp_line(siblc, psiflx, thtflx, vbelc, tblc, thtvlcx, psivlcx, grav):
    tfl = mat2tr(psiflx * RAD, thtflx * RAD)
    polar = polar_from_cart(siblc)
    ip_sltrange = float(polar[0])
    tol = mat2tr(float(polar[1]), float(polar[2]))
    vbeo = tol @ vbelc
    vbef = tfl @ vbelc
    nl_gain = NL_GAIN_FACT * (1.0 - exp(-ip_sltrange / DECREMENT))
    algv = np.array(
        [
            grav * sin(thtvlcx * RAD),
            LINE_GAIN * (-vbeo[1] + nl_gain * vbef[1]),
            LINE_GAIN * (-vbeo[2] + nl_gain * vbef[2]) - grav * cos(thtvlcx * RAD),
        ],
        dtype=float,
    )
    tvl = mat2tr(psivlcx * RAD, thtvlcx * RAD)
    tbv = tblc @ tvl.T
    acbx = tbv @ algv * (1.0 / AGRAV)
    return acbx, ip_sltrange, nl_gain, vbeo, vbef


def _cpp_term_pronav(tblc, grav_bias, gnav, tgnav, gn, gnd, int_step, ddab, psisb, thtsb, lamdqb, lamdrb):
    gravb = tblc @ np.array([0.0, 0.0, grav_bias], dtype=float)
    if tgnav:
        gnd_new = (gnav - gn) / tgnav
        gn = float(integrate(gnd_new, gnd, gn, int_step))
        gnd = gnd_new
    else:
        gn = gnav
    gnav_comp = -gn * ddab
    apnyx = gnav_comp * lamdrb / (cos(psisb) * AGRAV)
    apnzx = (
        gnav_comp
        * (lamdrb * tan(thtsb) * tan(psisb) + lamdqb / cos(thtsb))
        / AGRAV
    )
    allx = apnyx - gravb[1]
    annx = apnzx + gravb[2]
    acbx = np.array([0.0, allx, -annx], dtype=float)
    return acbx, gn, gnd, gnav_comp, apnyx, apnzx


def _cpp_term_comp(
    tblc,
    grav_bias,
    gnav,
    tgnav,
    gn,
    gnd,
    int_step,
    sbel,
    vbel,
    stel,
    vtel,
    thtpb,
    psipb,
    sigdy,
    sigdz,
    fspcb,
):
    sbtl = sbel - stel
    dbt = float(np.linalg.norm(sbtl))
    dcvel = float(sbtl @ (vbel - vtel)) / dbt
    fspcb1 = float(fspcb[0])
    adely = fspcb1 * tan(psipb) / AGRAV
    adelz = fspcb1 * tan(thtpb) / (cos(psipb) * AGRAV)
    gravb = tblc @ np.array([0.0, 0.0, grav_bias], dtype=float)
    if tgnav:
        gnd_new = (gnav - gn) / tgnav
        gn = float(integrate(gnd_new, gnd, gn, int_step))
        gnd = gnd_new
    else:
        gn = gnav
    gnav_comp = -gn * dcvel
    apnyx = gnav_comp * sigdz / (cos(psipb) * AGRAV)
    apnzx = (
        gnav_comp
        * (sigdz * tan(thtpb) * tan(psipb) + sigdy / cos(thtpb))
        / AGRAV
    )
    allx = apnyx + adely - gravb[1]
    annx = apnzx + adelz + gravb[2]
    acbx = np.array([0.0, allx, -annx], dtype=float)
    return acbx, gn, gnd, gnav_comp, apnyx, apnzx, adely, adelz, dcvel


def _plant_gmax(store, gmax=GMAX):
    store.define(Field("gmax", gmax, "real", "diag", "aerodynamics", ("plot",)))


def _plant_line(store, *, siel=SIEL_NORTH, tblc=IDENTITY, vbelc=VBELC, sbelc=SBELC):
    _plant_gmax(store)
    store.define(Field("TBLC", tblc, "mat", "out", "ins"))
    store.define(Field("VBELC", vbelc, "vec", "out", "ins"))
    store.define(Field("SBELC", sbelc, "vec", "out", "ins"))
    store.define(Field("thtvlcx", 0.0, "real", "out", "ins", ("plot",)))
    store.define(Field("psivlcx", 0.0, "real", "out", "ins", ("plot",)))
    store.define(Field("grav", GRAV, "real", "out", "environment"))
    store.define(Field("sbel1", 0.0, "real", "data", "newton"))
    store.define(Field("sbel2", 0.0, "real", "data", "newton"))
    store.define(Field("sbel3", 0.0, "real", "data", "newton"))
    store.set("line_gain", LINE_GAIN)
    store.set("nl_gain_fact", NL_GAIN_FACT)
    store.set("decrement", DECREMENT)
    store.set("thtflx", THTFLX)
    store.set("mguide", 20)
    return siel


def _plant_term7(store):
    _plant_gmax(store)
    store.define(Field("TBLC", IDENTITY, "mat", "out", "ins"))
    store.define(Field("ddab", -400.0, "real", "out", "sensor", ("plot",)))
    store.define(Field("psisb", 0.05, "real", "state", "sensor", ("plot",)))
    store.define(Field("thtsb", -0.02, "real", "state", "sensor", ("plot",)))
    store.define(Field("lamdqb", 0.01, "real", "out", "sensor", ("plot",)))
    store.define(Field("lamdrb", -0.015, "real", "out", "sensor", ("plot",)))
    store.set("mguide", 7)
    store.set("gnav", GNAV)
    store.set("tgnav", 0.0)


def _plant_term6(store):
    _plant_gmax(store)
    store.define(Field("TBLC", IDENTITY, "mat", "out", "ins"))
    store.define(Field("FSPCB", (12.0, 0.2, -9.5), "vec", "out", "ins"))
    store.define(Field("SBEL", (0.0, 0.0, -1000.0), "vec", "state", "newton"))
    store.define(Field("VBEL", (16.0, 4.0, -2.0), "vec", "out", "newton"))
    store.define(Field("STEL", (800.0, 100.0, -900.0), "vec", "out", "sensor"))
    store.define(Field("VTEL", (0.0, 250.0, 0.0), "vec", "out", "sensor"))
    store.define(Field("thtpb", 0.03, "real", "out", "sensor"))
    store.define(Field("psipb", -0.04, "real", "out", "sensor"))
    store.define(Field("sigdy", 0.012, "real", "out", "sensor", ("plot",)))
    store.define(Field("sigdz", -0.008, "real", "out", "sensor", ("plot",)))
    store.set("mguide", 6)
    store.set("gnav", GNAV_IR)
    store.set("tgnav", TGNAV)


def _radar(siel1, siel2=None, siel3=None, name="R"):
    vars_ = {"SIEL1": np.asarray(siel1, dtype=float)}
    if siel2 is not None:
        vars_["SIEL2"] = np.asarray(siel2, dtype=float)
    if siel3 is not None:
        vars_["SIEL3"] = np.asarray(siel3, dtype=float)
    return Packet(name=name, type="RADAR0", status=1, vars=vars_)


def _ready_line(siel=SIEL_NORTH, combus=None, vehicle_slot=0):
    vehicle = _Vehicle()
    guid = Sam6Guidance()
    guid.define(vehicle)
    _plant_line(vehicle.store, siel=siel)
    if combus is None:
        combus = [
            Packet(name="SAM", type="MISSILE6", status=1, vars={}),
            _radar(siel),
        ]
        vehicle_slot = 0
    return vehicle, guid, _ctx(combus=combus, vehicle_slot=vehicle_slot)


def test_name_is_guidance():
    assert Sam6Guidance().name == "guidance"


def test_define_cpp_fields():
    vehicle = _Vehicle()
    Sam6Guidance().define(vehicle)
    store = vehicle.store
    assert tuple(store.names()) == DEFINED
    for name in DEFINED:
        field = store.field(name)
        assert field.module == "guidance"
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        if name in INT_FIELDS:
            assert field.type == "int"
            assert store.get(name) == 0
        elif name in VEC_FIELDS:
            assert field.type == "vec"
            np.testing.assert_allclose(store.get(name), np.zeros(3), atol=ATOL)
        elif name == "grav_bias":
            assert field.type == "real"
            assert _approx(store.get(name), 1.0)
        else:
            assert field.type == "real"
            assert _approx(store.get(name), 0.0), name


def test_define_does_not_register_ins_newton_sensor_or_gmax():
    vehicle = _Vehicle()
    Sam6Guidance().define(vehicle)
    for name in NOT_DEFINED:
        assert name not in vehicle.store.names()


def test_mguide_0_writes_zero_commands():
    vehicle = _Vehicle()
    guid = Sam6Guidance()
    guid.define(vehicle)
    _plant_gmax(vehicle.store)
    vehicle.store.set("ancomx", 9.0)
    vehicle.store.set("alcomx", -3.0)
    guid.execute(vehicle, _ctx())
    assert _approx(vehicle.store.get("ancomx"), 0.0)
    assert _approx(vehicle.store.get("alcomx"), 0.0)
    assert _approx(vehicle.store.get("allx"), 0.0)
    assert _approx(vehicle.store.get("annx"), 0.0)


def test_mguide_20_radar_siel1_1km_north_finite_commands():
    vehicle, guid, ctx = _ready_line()
    guid.execute(vehicle, ctx)
    ancomx = vehicle.store.get("ancomx")
    alcomx = vehicle.store.get("alcomx")
    assert np.isfinite(ancomx)
    assert np.isfinite(alcomx)
    assert abs(ancomx) + abs(alcomx) > 0.0


def test_mguide_20_line_vs_cpp_replica():
    vehicle, guid, ctx = _ready_line()
    siblc = SIEL_NORTH - SBELC
    sibl0 = SIEL_NORTH - np.array([0.0, 0.0, 0.0])
    psiflx = float(polar_from_cart(sibl0)[1]) * DEG
    acbx, ip_sltrange, nl_gain, vbeo, vbef = _cpp_line(
        siblc, psiflx, THTFLX, VBELC, np.array(IDENTITY, dtype=float), 0.0, 0.0, GRAV
    )
    alcomx, ancomx = _circular_limit(float(acbx[1]), float(-acbx[2]), GMAX)
    guid.execute(vehicle, ctx)
    store = vehicle.store
    assert _approx(store.get("ancomx"), ancomx)
    assert _approx(store.get("alcomx"), alcomx)
    assert _approx(store.get("psiflx"), psiflx)
    assert _approx(store.get("ip_sltrange"), ip_sltrange)
    assert _approx(store.get("nl_gain"), nl_gain)
    np.testing.assert_allclose(store.get("SIBLC"), siblc, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VBEO"), vbeo, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VBEF"), vbef, rtol=RTOL, atol=ATOL)
    assert _approx(store.get("allx"), float(acbx[1]))
    assert _approx(store.get("annx"), float(-acbx[2]))


def test_mguide_30_raises():
    vehicle = _Vehicle()
    guid = Sam6Guidance()
    guid.define(vehicle)
    _plant_gmax(vehicle.store)
    vehicle.store.set("mguide", 30)
    with pytest.raises(ValueError, match="guid_mid"):
        guid.execute(vehicle, _ctx())


def test_mguide_7_seeker_kinematics_finite_commands():
    vehicle = _Vehicle()
    guid = Sam6Guidance()
    guid.define(vehicle)
    _plant_term7(vehicle.store)
    guid.execute(vehicle, _ctx())
    assert np.isfinite(vehicle.store.get("ancomx"))
    assert np.isfinite(vehicle.store.get("alcomx"))
    assert abs(vehicle.store.get("ancomx")) + abs(vehicle.store.get("alcomx")) > 0.0


def test_mguide_7_pronav_vs_cpp_replica():
    vehicle = _Vehicle()
    guid = Sam6Guidance()
    guid.define(vehicle)
    _plant_term7(vehicle.store)
    tblc = np.array(IDENTITY, dtype=float)
    acbx, gn, gnd, gnav_comp, apnyx, apnzx = _cpp_term_pronav(
        tblc, GRAV_BIAS, GNAV, 0.0, 0.0, 0.0, DT, -400.0, 0.05, -0.02, 0.01, -0.015
    )
    alcomx, ancomx = _circular_limit(float(acbx[1]), float(-acbx[2]), GMAX)
    guid.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("ancomx"), ancomx)
    assert _approx(store.get("alcomx"), alcomx)
    assert _approx(store.get("gn"), gn)
    assert _approx(store.get("gnav_comp"), gnav_comp)
    assert _approx(store.get("apnyx"), apnyx)
    assert _approx(store.get("apnzx"), apnzx)


def test_mguide_6_comp_pronav_vs_cpp_replica():
    vehicle = _Vehicle()
    guid = Sam6Guidance()
    guid.define(vehicle)
    _plant_term6(vehicle.store)
    tblc = np.array(IDENTITY, dtype=float)
    acbx, gn, gnd, gnav_comp, apnyx, apnzx, adely, adelz, dcvel = _cpp_term_comp(
        tblc,
        GRAV_BIAS,
        GNAV_IR,
        TGNAV,
        0.0,
        0.0,
        DT,
        np.array([0.0, 0.0, -1000.0]),
        np.array([16.0, 4.0, -2.0]),
        np.array([800.0, 100.0, -900.0]),
        np.array([0.0, 250.0, 0.0]),
        0.03,
        -0.04,
        0.012,
        -0.008,
        np.array([12.0, 0.2, -9.5]),
    )
    alcomx, ancomx = _circular_limit(float(acbx[1]), float(-acbx[2]), GMAX)
    guid.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("ancomx"), ancomx)
    assert _approx(store.get("alcomx"), alcomx)
    assert _approx(store.get("gn"), gn)
    assert _approx(store.get("gnd"), gnd)
    assert _approx(store.get("gnav_comp"), gnav_comp)
    assert _approx(store.get("apnyx"), apnyx)
    assert _approx(store.get("apnzx"), apnzx)
    assert _approx(store.get("adely"), adely)
    assert _approx(store.get("adelz"), adelz)
    assert _approx(store.get("dcvel"), dcvel)


def test_identifies_radar_by_type_not_cpp_id():
    decoy = Packet(
        name="f1",
        type="AIRCRAFT3",
        status=1,
        vars={"SIEL1": np.array([9.0e6, 0.0, 0.0])},
    )
    radar = _radar(SIEL_NORTH, name="other")
    missile = Packet(name="m1", type="MISSILE6", status=1, vars={})
    vehicle, guid, ctx = _ready_line(
        combus=[decoy, radar, missile], vehicle_slot=2
    )
    guid.execute(vehicle, ctx)
    np.testing.assert_allclose(
        vehicle.store.get("SIBLC"), SIEL_NORTH - SBELC, rtol=RTOL, atol=ATOL
    )


def test_ip_indexes_among_missile6_not_combus_slot():
    radar = _radar(
        SIEL_NORTH,
        siel2=np.array([2000.0, 0.0, 0.0]),
        siel3=np.array([3000.0, 0.0, 0.0]),
    )
    combus = [
        radar,
        Packet(name="a1", type="AIRCRAFT3", status=1, vars={}),
        Packet(name="M1", type="MISSILE6", status=1, vars={}),
    ]
    vehicle, guid, ctx = _ready_line(combus=combus, vehicle_slot=2)
    guid.execute(vehicle, ctx)
    np.testing.assert_allclose(
        vehicle.store.get("SIBLC"), SIEL_NORTH - SBELC, rtol=RTOL, atol=ATOL
    )
    assert not np.allclose(vehicle.store.get("SIBLC"), [3000.0, 0.0, 0.0])


def test_second_missile_uses_siel2():
    siel2 = np.array([2000.0, 50.0, 0.0], dtype=float)
    radar = _radar(SIEL_NORTH, siel2=siel2)
    combus = [
        Packet(name="M1", type="MISSILE6", status=1, vars={}),
        Packet(name="M2", type="MISSILE6", status=1, vars={}),
        radar,
    ]
    vehicle, guid, ctx = _ready_line(combus=combus, vehicle_slot=1)
    guid.execute(vehicle, ctx)
    np.testing.assert_allclose(
        vehicle.store.get("SIBLC"), siel2 - SBELC, rtol=RTOL, atol=ATOL
    )


def test_circular_limiter_caps_at_gmax():
    vehicle = _Vehicle()
    guid = Sam6Guidance()
    guid.define(vehicle)
    _plant_line(vehicle.store)
    vehicle.store.set("gmax", 0.05)
    combus = [
        Packet(name="SAM", type="MISSILE6", status=1, vars={}),
        _radar(SIEL_NORTH),
    ]
    guid.execute(vehicle, _ctx(combus=combus, vehicle_slot=0))
    alcomx = vehicle.store.get("alcomx")
    ancomx = vehicle.store.get("ancomx")
    aa = sqrt(alcomx * alcomx + ancomx * ancomx)
    assert _approx(aa, 0.05)
    siblc = SIEL_NORTH - SBELC
    sibl0 = SIEL_NORTH
    psiflx = float(polar_from_cart(sibl0)[1]) * DEG
    acbx, *_ = _cpp_line(
        siblc, psiflx, THTFLX, VBELC, np.array(IDENTITY, dtype=float), 0.0, 0.0, GRAV
    )
    want_al, want_an = _circular_limit(float(acbx[1]), float(-acbx[2]), 0.05)
    assert _approx(alcomx, want_al)
    assert _approx(ancomx, want_an)


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.flat6.sam6.guidance as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "Plane5" not in src
    assert "Plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src


def test_initialize_and_terminate_are_pass():
    vehicle = _Vehicle()
    guid = Sam6Guidance()
    guid.define(vehicle)
    _plant_gmax(vehicle.store)
    ctx = _ctx()
    assert guid.initialize(vehicle, ctx) is None
    assert guid.terminate(vehicle, ctx) is None
    guid.execute(vehicle, ctx)
    assert _approx(vehicle.store.get("ancomx"), 0.0)

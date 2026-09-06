from math import acos, cos, sin, sqrt

import numpy as np
import pytest

from cadac.constants import DEG, RAD, REARTH
from cadac.kernel.combus import Packet
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.math.earth import cadtei, cadtge
from cadac.math.frames import cadtbv, mat2tr, polar_from_cart
from cadac.vehicles.hyper5.seeker import Hyper5Seeker

# Demo 4.7 Terminal pro-nav with seeker
LONX = -106.28
LATX = 33.35
ALT = 2400.0
PSIVGX = 0.0
THTVGX = -10.5
DVBE = 254.0
ALPHAX = -1.5
PHIMVX = 0.0
ACQ_RANGE = 6000.0
TIME = 0.0

TGT_LONX = -106.28
TGT_LATX = 33.4
TGT_ALT = 1200.0
TGT_PSIVGX = 0.0
TGT_THTVGX = 0.0
TGT_DVBE = 0.0

RTOL = 1e-12
ATOL = 1e-14

SEEKER_FIELDS = {
    "mseeker": ("int", "data/save", ("scrn",)),
    "acq_range": ("real", "data", ()),
    "range_go": ("real", "out", ("plot", "scrn")),
    "STBG": ("vec", "out", ("plot",)),
    "WOEB": ("vec", "out", ()),
    "closing_speed": ("real", "out", ()),
    "time_go": ("real", "out", ("plot", "scrn")),
    "psisbx": ("real", "out", ("plot", "scrn")),
    "thtsbx": ("real", "out", ("plot", "scrn")),
    "targ_com_slot": ("int", "save", ()),
    "UTBB": ("vec", "out", ()),
    "acquisition": ("int", "init/save", ("scrn",)),
}

NOT_DEFINED = (
    "mcontrol",
    "mguidance",
    "TBG",
    "TIG",
    "VBEG",
    "SBII",
    "time",
    "tig",
    "vbeg",
    "sbii",
    "lonx",
    "latx",
    "alt",
    "psivgx",
    "thtvgx",
    "dvbe",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _skew(vec):
    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def _round3_kinematics(lonx, latx, alt, psivgx, thtvgx, dvbe, time):
    sbig = np.array([0.0, 0.0, -(alt + REARTH)])
    tge = cadtge(lonx * RAD, latx * RAD)
    teg = tge.T
    sbie = teg @ sbig
    tei = cadtei(time)
    sbii = tei.T @ sbie
    psivg = psivgx * RAD
    thtvg = thtvgx * RAD
    vbeg = np.array(
        [
            dvbe * np.cos(thtvg) * np.cos(psivg),
            dvbe * np.cos(thtvg) * np.sin(psivg),
            dvbe * (-np.sin(thtvg)),
        ]
    )
    tig = tei.T @ teg
    return tig, sbii, vbeg


def _ground_range(lonx_c, latx_c, lonx_t, latx_t):
    lon_c = lonx_c * RAD
    lat_c = latx_c * RAD
    lon_t = lonx_t * RAD
    lat_t = latx_t * RAD
    dum = sin(lat_t) * sin(lat_c) + cos(lat_t) * cos(lat_c) * cos(lon_t - lon_c)
    return REARTH * acos(dum)


def _expected_track(tig, sbii, vbeg, tbg, tgt_sbii, tgt_vbeg):
    stbi = tgt_sbii - sbii
    tgi = tig.T
    stbg = tgi @ stbi
    range_go = sqrt(float(stbg[0] ** 2 + stbg[1] ** 2 + stbg[2] ** 2))
    inv_dtb = 1.0 / range_go
    utbg = stbg * inv_dtb
    vtbg = tgt_vbeg - vbeg
    woeb = tbg @ _skew(utbg) @ vtbg * inv_dtb
    vbtg = vtbg * (-1.0)
    closing_speed = float(utbg @ vbtg)
    time_go = range_go / closing_speed
    utbb = tbg @ utbg
    polar = polar_from_cart(utbb)
    psisbx = polar[1] * DEG
    thtsbx = polar[2] * DEG
    return stbg, range_go, woeb, closing_speed, time_go, psisbx, thtsbx, utbb


def _plant_hyper(store, tig, sbii, vbeg, tbg, lonx=LONX, latx=LATX, time=TIME):
    store.define(Field("time", time, "real", "exec", "environment"))
    store.define(Field("tig", tig, "mat", "init/out", "newton"))
    store.define(Field("vbeg", vbeg, "vec", "state", "newton"))
    store.define(Field("sbii", sbii, "vec", "state", "newton"))
    store.define(Field("TBG", tbg, "mat", "out", "control"))
    store.define(Field("lonx", lonx, "real", "init/diag", "newton"))
    store.define(Field("latx", latx, "real", "init/diag", "newton"))


def _target_packet(lonx, latx, alt, psivgx, thtvgx, dvbe, vbeg, sbii, name="Truck"):
    return Packet(
        name=name,
        type="TARGET3",
        status=1,
        vars={
            "lonx": lonx,
            "latx": latx,
            "alt": alt,
            "psivgx": psivgx,
            "thtvgx": thtvgx,
            "dvbe": dvbe,
            "vbeg": vbeg,
            "sbii": sbii,
        },
    )


def _hyper_packet(lonx=0.0, latx=0.0):
    return Packet(
        name="RR3X",
        type="HYPER5",
        status=1,
        vars={"lonx": lonx, "latx": latx},
    )


def _ctx(combus, vehicle_slot=0):
    return SimContext(
        sim_time=0.0,
        int_step=0.05,
        event_time=0.0,
        out_fact=0.0,
        combus=combus,
        vehicle_slot=vehicle_slot,
    )


def _demo47():
    tig, sbii, vbeg = _round3_kinematics(
        LONX, LATX, ALT, PSIVGX, THTVGX, DVBE, TIME
    )
    _, tgt_sbii, tgt_vbeg = _round3_kinematics(
        TGT_LONX, TGT_LATX, TGT_ALT, TGT_PSIVGX, TGT_THTVGX, TGT_DVBE, TIME
    )
    tbg = cadtbv(PHIMVX * RAD, ALPHAX * RAD) @ mat2tr(PSIVGX * RAD, THTVGX * RAD)
    return tig, sbii, vbeg, tbg, tgt_sbii, tgt_vbeg


def _ready(
    mseeker=1,
    acq_range=ACQ_RANGE,
    acquisition=0,
    targ_com_slot=0,
    plant=True,
    combus=None,
    lonx=LONX,
    latx=LATX,
):
    vehicle = _Vehicle()
    seeker = Hyper5Seeker()
    seeker.define(vehicle)
    store = vehicle.store
    store.set("mseeker", mseeker)
    store.set("acq_range", acq_range)
    store.set("acquisition", acquisition)
    store.set("targ_com_slot", targ_com_slot)
    tig, sbii, vbeg, tbg, tgt_sbii, tgt_vbeg = _demo47()
    if plant:
        _plant_hyper(store, tig, sbii, vbeg, tbg, lonx=lonx, latx=latx)
    if combus is None:
        combus = [
            _hyper_packet(lonx=0.0, latx=90.0),
            _target_packet(
                TGT_LONX,
                TGT_LATX,
                TGT_ALT,
                TGT_PSIVGX,
                TGT_THTVGX,
                TGT_DVBE,
                tgt_vbeg,
                tgt_sbii,
            ),
        ]
    return vehicle, seeker, _ctx(combus), tig, sbii, vbeg, tbg, tgt_sbii, tgt_vbeg


def test_name_is_seeker():
    assert Hyper5Seeker().name == "seeker"


def test_define_registers_def_seeker_fields():
    vehicle = _Vehicle()
    Hyper5Seeker().define(vehicle)
    store = vehicle.store
    assert list(SEEKER_FIELDS) == store.names()
    for name, (ftype, role, outputs) in SEEKER_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "seeker"
        assert field.outputs == outputs
        if ftype == "int":
            assert store.get(name) == 0
            assert type(store.get(name)) is int
        elif ftype == "real":
            assert store.get(name) == 0.0
        else:
            np.testing.assert_array_equal(store.get(name), np.zeros(3))


def test_define_does_not_register_control_guidance_or_plant():
    vehicle = _Vehicle()
    Hyper5Seeker().define(vehicle)
    store = vehicle.store
    for name in NOT_DEFINED:
        with pytest.raises(KeyError):
            store.get(name)


def test_initialize_is_pass():
    vehicle = _Vehicle()
    seeker = Hyper5Seeker()
    seeker.define(vehicle)
    seeker.initialize(vehicle, _ctx([]))
    for name in SEEKER_FIELDS:
        if SEEKER_FIELDS[name][0] == "vec":
            np.testing.assert_array_equal(vehicle.store.get(name), np.zeros(3))
        elif SEEKER_FIELDS[name][0] == "int":
            assert vehicle.store.get(name) == 0
        else:
            assert vehicle.store.get(name) == 0.0


def test_terminate_exists_and_is_pass():
    vehicle = _Vehicle()
    seeker = Hyper5Seeker()
    seeker.define(vehicle)
    store = vehicle.store
    store.set("mseeker", 1)
    store.set("acq_range", 6000.0)
    store.set("range_go", 123.0)
    store.set("acquisition", 1)
    seeker.terminate(vehicle, _ctx([]))
    assert store.get("mseeker") == 1
    assert store.get("acq_range") == 6000.0
    assert store.get("range_go") == 123.0
    assert store.get("acquisition") == 1


def test_demo47_ground_range_is_inside_acq_range():
    range_m = _ground_range(LONX, LATX, TGT_LONX, TGT_LATX)
    assert range_m < ACQ_RANGE
    assert range_m == pytest.approx(5559.7, rel=1e-3)


def test_mseeker_0_does_not_write_track_outputs():
    vehicle, seeker, ctx, *_ = _ready(mseeker=0)
    store = vehicle.store
    store.set("range_go", 123.0)
    store.set("time_go", 4.0)
    store.set("psisbx", 9.0)
    store.set("thtsbx", -3.0)
    store.set("closing_speed", 77.0)
    store.set("targ_com_slot", 7)
    store.set("acquisition", 1)
    stbg = np.array([1.0, 2.0, 3.0])
    woeb = np.array([0.1, 0.2, 0.3])
    utbb = np.array([0.4, 0.5, 0.6])
    store.set("STBG", stbg)
    store.set("WOEB", woeb)
    store.set("UTBB", utbb)
    seeker.execute(vehicle, ctx)
    assert store.get("mseeker") == 0
    assert store.get("range_go") == 123.0
    assert store.get("time_go") == 4.0
    assert store.get("psisbx") == 9.0
    assert store.get("thtsbx") == -3.0
    assert store.get("closing_speed") == 77.0
    assert store.get("targ_com_slot") == 7
    assert store.get("acquisition") == 1
    np.testing.assert_array_equal(store.get("STBG"), stbg)
    np.testing.assert_array_equal(store.get("WOEB"), woeb)
    np.testing.assert_array_equal(store.get("UTBB"), utbb)


@pytest.mark.parametrize("mseeker", (2, 99, -1))
def test_unknown_mseeker_raises_valueerror(mseeker):
    vehicle, seeker, ctx, *_ = _ready(mseeker=mseeker)
    with pytest.raises(ValueError):
        seeker.execute(vehicle, ctx)


def test_mseeker_1_range_inside_6000_acquires_and_tracks():
    vehicle, seeker, ctx, tig, sbii, vbeg, tbg, tgt_sbii, tgt_vbeg = _ready(
        mseeker=1
    )
    store = vehicle.store
    store.define(Field("mcontrol", 44, "int", "data", "control"))
    store.define(Field("mguidance", 66, "int", "data", "guidance"))
    assert _ground_range(LONX, LATX, TGT_LONX, TGT_LATX) < ACQ_RANGE
    seeker.execute(vehicle, ctx)
    assert store.get("mseeker") == 3
    assert store.get("acquisition") == 1
    assert store.get("targ_com_slot") == 1
    assert store.get("mcontrol") == 44
    assert store.get("mguidance") == 66
    expected = _expected_track(tig, sbii, vbeg, tbg, tgt_sbii, tgt_vbeg)
    stbg, range_go, woeb, closing_speed, time_go, psisbx, thtsbx, utbb = expected
    np.testing.assert_allclose(store.get("STBG"), stbg, rtol=RTOL, atol=ATOL)
    assert _approx(store.get("range_go"), range_go)
    np.testing.assert_allclose(store.get("WOEB"), woeb, rtol=RTOL, atol=ATOL)
    assert _approx(store.get("closing_speed"), closing_speed)
    assert _approx(store.get("time_go"), time_go)
    assert _approx(store.get("psisbx"), psisbx)
    assert _approx(store.get("thtsbx"), thtsbx)
    np.testing.assert_allclose(store.get("UTBB"), utbb, rtol=RTOL, atol=ATOL)
    assert np.all(np.isfinite(store.get("STBG")))
    assert np.all(np.isfinite(store.get("WOEB")))
    assert np.all(np.isfinite(store.get("UTBB")))


def test_mseeker_3_tracks_saved_target_slot():
    tig, sbii, vbeg, tbg, tgt_sbii, tgt_vbeg = _demo47()
    vehicle, seeker, ctx, *_ = _ready(
        mseeker=3, acquisition=1, targ_com_slot=1
    )
    seeker.execute(vehicle, ctx)
    expected = _expected_track(tig, sbii, vbeg, tbg, tgt_sbii, tgt_vbeg)
    stbg, range_go, woeb, closing_speed, time_go, psisbx, thtsbx, utbb = expected
    store = vehicle.store
    assert store.get("mseeker") == 3
    assert store.get("targ_com_slot") == 1
    np.testing.assert_allclose(store.get("STBG"), stbg, rtol=RTOL, atol=ATOL)
    assert _approx(store.get("range_go"), range_go)
    np.testing.assert_allclose(store.get("WOEB"), woeb, rtol=RTOL, atol=ATOL)
    assert _approx(store.get("closing_speed"), closing_speed)
    np.testing.assert_allclose(store.get("UTBB"), utbb, rtol=RTOL, atol=ATOL)


def test_mseeker_1_out_of_range_does_not_acquire_or_write_track():
    vehicle, seeker, ctx, *_ = _ready(mseeker=1, latx=LATX - 1.0)
    store = vehicle.store
    store.set("range_go", 999.0)
    stbg = np.array([8.0, 7.0, 6.0])
    store.set("STBG", stbg)
    seeker.execute(vehicle, ctx)
    assert store.get("mseeker") == 1
    assert store.get("acquisition") == 0
    assert store.get("targ_com_slot") == 0
    assert store.get("range_go") == 999.0
    np.testing.assert_array_equal(store.get("STBG"), stbg)


def test_identifies_target_by_type_not_id_prefix():
    tig, sbii, vbeg, tbg, tgt_sbii, tgt_vbeg = _demo47()
    combus = [
        Packet(name="t1", type="HYPER5", status=1, vars={"lonx": TGT_LONX, "latx": TGT_LATX}),
        Packet(
            name="s1",
            type="SATELLITE3",
            status=1,
            vars={
                "lonx": TGT_LONX,
                "latx": TGT_LATX,
                "vbeg": tgt_vbeg,
                "sbii": tgt_sbii,
            },
        ),
        _target_packet(
            TGT_LONX,
            TGT_LATX,
            TGT_ALT,
            TGT_PSIVGX,
            TGT_THTVGX,
            TGT_DVBE,
            tgt_vbeg,
            tgt_sbii,
            name="Truck",
        ),
    ]
    vehicle, seeker, ctx, *_ = _ready(mseeker=1, combus=combus)
    seeker.execute(vehicle, ctx)
    store = vehicle.store
    assert store.get("mseeker") == 3
    assert store.get("targ_com_slot") == 2
    expected = _expected_track(tig, sbii, vbeg, tbg, tgt_sbii, tgt_vbeg)
    np.testing.assert_allclose(store.get("STBG"), expected[0], rtol=RTOL, atol=ATOL)


def test_own_lonx_latx_from_hyper_store_not_combus():
    vehicle, seeker, ctx, *_ = _ready(mseeker=1)
    assert ctx.combus[0].vars["lonx"] == 0.0
    assert ctx.combus[0].vars["latx"] == 90.0
    seeker.execute(vehicle, ctx)
    assert vehicle.store.get("mseeker") == 3


def test_packet_kinematics_use_lowercase_vbeg_sbii():
    tig, sbii, vbeg, tbg, tgt_sbii, tgt_vbeg = _demo47()
    packet = _target_packet(
        TGT_LONX,
        TGT_LATX,
        TGT_ALT,
        TGT_PSIVGX,
        TGT_THTVGX,
        TGT_DVBE,
        tgt_vbeg,
        tgt_sbii,
    )
    assert "vbeg" in packet.vars
    assert "sbii" in packet.vars
    assert "VBEG" not in packet.vars
    assert "SBII" not in packet.vars
    vehicle, seeker, ctx, *_ = _ready(
        mseeker=3, acquisition=1, targ_com_slot=1, combus=[_hyper_packet(), packet]
    )
    seeker.execute(vehicle, ctx)
    expected = _expected_track(tig, sbii, vbeg, tbg, tgt_sbii, tgt_vbeg)
    np.testing.assert_allclose(vehicle.store.get("UTBB"), expected[-1], rtol=RTOL, atol=ATOL)

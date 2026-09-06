from math import atan2, cos, exp, sin, sqrt

import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.aim5.guidance import SMALL, Aim5Guidance

RTOL = 1e-12
ATOL = 1e-14

WOEA = np.array([0.0, 0.01, -0.02])
UTAA = np.array([0.9, 0.1, 0.4])
DVTA = -300.0
GRAV = 9.8
GMAX = 40.0
MGUID = 1
GNAV = 4.0

GUIDANCE_FIELDS = {
    "mguid": ("int", "data", ()),
    "gnav": ("real", "data", ()),
    "tgo_manvr": ("real", "data", ()),
    "amp_manvr": ("real", "data", ()),
    "frq_manvr": ("real", "data", ()),
    "tgo63_manvr": ("real", "data", ()),
    "annx": ("real", "diag", ()),
    "allx": ("real", "diag", ()),
    "an_manvr": ("real", "diag", ()),
    "al_manvr": ("real", "diag", ()),
    "ancomx": ("real", "out", ("scrn", "plot")),
    "alcomx": ("real", "out", ("scrn", "plot")),
}

EXTERNALS = (
    "dvta",
    "tgo_aim",
    "UTAA",
    "WOEA",
    "gmax",
    "grav",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


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


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.002,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _expected_guidance(
    *,
    mguid,
    gnav,
    woea,
    utaa,
    dvta,
    grav,
    gmax,
    tgo_aim,
    tgo_manvr,
    amp_manvr,
    frq_manvr,
    tgo63_manvr,
):
    # Replica of Aim::guidance (last write of aim[108] is an_manvr, not amp)
    guid_manvr = mguid // 10
    guid_mode = mguid % 10
    if guid_mode not in {0, 1} or guid_manvr not in {0, 1}:
        raise ValueError(
            f"unsupported mguid={mguid}: guid_mode={guid_mode} guid_manvr={guid_manvr}"
        )
    annx = 0.0
    allx = 0.0
    an_manvr = 0.0
    al_manvr = 0.0
    if guid_mode == 1:
        apna = _skew(woea) @ utaa * (gnav * abs(dvta))
        annx = -apna[2] / grav
        allx = apna[1] / grav
    if guid_manvr == 1 and tgo_aim < tgo_manvr:
        amp = amp_manvr * (1.0 - exp(-tgo_aim / tgo63_manvr))
        an_manvr = amp * sin(frq_manvr * tgo_aim)
        al_manvr = amp * cos(frq_manvr * tgo_aim)
        annx += an_manvr
        allx += al_manvr
    aax = sqrt(allx**2 + annx**2)
    if aax > gmax:
        aax = gmax
    if abs(annx) < SMALL or abs(allx) < SMALL:
        phi = 0.0
    else:
        phi = atan2(annx, allx)
    alcomx = aax * cos(phi)
    ancomx = aax * sin(phi)
    return {
        "annx": annx,
        "allx": allx,
        "an_manvr": an_manvr,
        "al_manvr": al_manvr,
        "ancomx": ancomx,
        "alcomx": alcomx,
        "amp": (
            amp_manvr * (1.0 - exp(-tgo_aim / tgo63_manvr))
            if guid_manvr == 1 and tgo_aim < tgo_manvr
            else 0.0
        ),
    }


def _ready(
    *,
    mguid=MGUID,
    gnav=GNAV,
    woea=None,
    utaa=None,
    dvta=DVTA,
    grav=GRAV,
    gmax=GMAX,
    tgo_aim=100.0,
    tgo_manvr=0.0,
    amp_manvr=0.0,
    frq_manvr=0.0,
    tgo63_manvr=0.0,
):
    if woea is None:
        woea = WOEA
    if utaa is None:
        utaa = UTAA
    vehicle = _Vehicle()
    guidance = Aim5Guidance()
    guidance.define(vehicle)
    store = vehicle.store
    store.define(Field("dvta", dvta, "real", "out", "seeker"))
    store.define(Field("tgo_aim", tgo_aim, "real", "diag", "seeker"))
    store.define(Field("UTAA", utaa, "vec", "out", "seeker"))
    store.define(Field("WOEA", woea, "vec", "out", "seeker"))
    store.define(Field("gmax", gmax, "real", "out", "aerodynamics"))
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.set("mguid", mguid)
    store.set("gnav", gnav)
    store.set("tgo_manvr", tgo_manvr)
    store.set("amp_manvr", amp_manvr)
    store.set("frq_manvr", frq_manvr)
    store.set("tgo63_manvr", tgo63_manvr)
    guidance.initialize(vehicle, _ctx())
    return vehicle, guidance


def test_name_is_guidance():
    assert Aim5Guidance().name == "guidance"
    assert SMALL == 1e-7


def test_define_registers_cpp_guidance_fields():
    vehicle = _Vehicle()
    Aim5Guidance().define(vehicle)
    store = vehicle.store
    plot = ("scrn", "plot")
    for name, (ftype, role, outputs) in GUIDANCE_FIELDS.items():
        field = store.field(name)
        assert field.type == ftype
        assert field.role == role
        assert field.module == "guidance"
        assert field.outputs == outputs
        if ftype == "int":
            assert field.value == 0
        else:
            assert field.value == pytest.approx(0.0, abs=ATOL)
    assert store.field("ancomx").outputs == plot
    assert store.field("alcomx").outputs == plot
    for name in EXTERNALS:
        assert name not in store.names()


def test_pronav_mguid_1_matches_replica():
    vehicle, guidance = _ready()
    want = _expected_guidance(
        mguid=MGUID,
        gnav=GNAV,
        woea=WOEA,
        utaa=UTAA,
        dvta=DVTA,
        grav=GRAV,
        gmax=GMAX,
        tgo_aim=100.0,
        tgo_manvr=0.0,
        amp_manvr=0.0,
        frq_manvr=0.0,
        tgo63_manvr=0.0,
    )

    guidance.execute(vehicle, _ctx())

    store = vehicle.store
    for name in ("ancomx", "alcomx", "annx", "allx"):
        np.testing.assert_allclose(store.get(name), want[name], rtol=RTOL, atol=ATOL)
    assert store.get("an_manvr") == pytest.approx(0.0, abs=ATOL)
    assert store.get("al_manvr") == pytest.approx(0.0, abs=ATOL)


def test_mguid_0_limiter_on_zeros():
    vehicle, guidance = _ready(mguid=0)

    guidance.execute(vehicle, _ctx())

    store = vehicle.store
    assert store.get("annx") == pytest.approx(0.0, abs=ATOL)
    assert store.get("allx") == pytest.approx(0.0, abs=ATOL)
    assert store.get("ancomx") == pytest.approx(0.0, abs=ATOL)
    assert store.get("alcomx") == pytest.approx(0.0, abs=ATOL)


def test_mguid_11_spiral_added_before_limiter():
    tgo_aim = 5.0
    tgo_manvr = 10.0
    amp_manvr = 2.0
    frq_manvr = 1.0
    tgo63_manvr = 3.0
    vehicle, guidance = _ready(
        mguid=11,
        tgo_aim=tgo_aim,
        tgo_manvr=tgo_manvr,
        amp_manvr=amp_manvr,
        frq_manvr=frq_manvr,
        tgo63_manvr=tgo63_manvr,
    )
    want = _expected_guidance(
        mguid=11,
        gnav=GNAV,
        woea=WOEA,
        utaa=UTAA,
        dvta=DVTA,
        grav=GRAV,
        gmax=GMAX,
        tgo_aim=tgo_aim,
        tgo_manvr=tgo_manvr,
        amp_manvr=amp_manvr,
        frq_manvr=frq_manvr,
        tgo63_manvr=tgo63_manvr,
    )
    amp = amp_manvr * (1.0 - exp(-tgo_aim / tgo63_manvr))
    assert want["an_manvr"] == pytest.approx(amp * sin(frq_manvr * tgo_aim), rel=RTOL, abs=ATOL)
    assert want["al_manvr"] == pytest.approx(amp * cos(frq_manvr * tgo_aim), rel=RTOL, abs=ATOL)
    assert abs(want["an_manvr"] - amp) > 1e-6

    guidance.execute(vehicle, _ctx())

    store = vehicle.store
    np.testing.assert_allclose(store.get("an_manvr"), want["an_manvr"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("al_manvr"), want["al_manvr"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("annx"), want["annx"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("allx"), want["allx"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("ancomx"), want["ancomx"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("alcomx"), want["alcomx"], rtol=RTOL, atol=ATOL)
    assert store.get("an_manvr") != pytest.approx(want["amp"], rel=RTOL, abs=ATOL)


def test_mguid_21_guid_manvr_2_raises():
    vehicle, guidance = _ready(mguid=21)
    with pytest.raises(ValueError):
        guidance.execute(vehicle, _ctx())


def test_mguid_2_guid_mode_2_raises():
    vehicle, guidance = _ready(mguid=2)
    with pytest.raises(ValueError):
        guidance.execute(vehicle, _ctx())


def test_initialize_and_terminate_are_pass():
    vehicle, guidance = _ready()
    guidance.terminate(vehicle, _ctx())
    assert vehicle.store.get("ancomx") == pytest.approx(0.0, abs=ATOL)
    assert vehicle.store.get("alcomx") == pytest.approx(0.0, abs=ATOL)

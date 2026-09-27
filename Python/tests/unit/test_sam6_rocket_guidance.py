from math import atan2, cos, exp, fabs, sin, sqrt
from pathlib import Path

import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.sam6.rocket import Sam6RocketGuidance

RTOL = 1e-12
ATOL = 1e-14
SMALL = 1e-7
GMAX = 10.0
GRAV = 10.0
GNAV = 4.0
DVTA = 5.0
WOEA = np.array([0.0, 0.2, 0.3], dtype=float)
UTAA = np.array([1.0, 0.0, 0.0], dtype=float)
WOEA_LATERAL_ONLY = np.array([0.0, 0.5, 0.0], dtype=float)
TGO_TGT = 5.0
TGO_MANVR = 10.0
AMP_MANVR = 2.0
FRQ_MANVR = 0.5
TGO63_MANVR = 4.0
ZEROS3 = (0.0, 0.0, 0.0)

DEFINED = (
    "mguide",
    "gnav",
    "tgo_manvr",
    "amp_manvr",
    "frq_manvr",
    "tgo63_manvr",
    "annx",
    "allx",
    "an_manvr",
    "al_manvr",
    "ancomx",
    "alcomx",
)
ROLES = {
    "mguide": "data",
    "gnav": "data",
    "tgo_manvr": "data",
    "amp_manvr": "data",
    "frq_manvr": "data",
    "tgo63_manvr": "data",
    "annx": "diag",
    "allx": "diag",
    "an_manvr": "diag",
    "al_manvr": "diag",
    "ancomx": "out",
    "alcomx": "out",
}
OUTPUTS = {name: () for name in DEFINED}
INT_FIELDS = ("mguide",)
NOT_DEFINED = (
    "grav",
    "alt",
    "gmax",
    "dvta",
    "tgo_tgt",
    "UTAA",
    "WOEA",
    "flag_exo",
    "alt_endo",
    "maut",
    "ancomx_bias",
    "SAEL",
    "VAEL",
    "TAL",
    "time",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.001,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _cpp_limit(allx, annx, gmax):
    aax = sqrt(allx * allx + annx * annx)
    if aax > gmax:
        aax = gmax
    if fabs(annx) < SMALL or fabs(allx) < SMALL:
        phi = 0.0
    else:
        phi = atan2(annx, allx)
    return aax * cos(phi), aax * sin(phi)


def _cpp_pronav(woea, utaa, gnav, dvta, grav):
    apna = np.cross(woea, utaa) * gnav * fabs(dvta)
    annx = -float(apna[2]) / grav
    allx = float(apna[1]) / grav
    return allx, annx


def _cpp_spiral(tgo_tgt, amp_manvr, frq_manvr, tgo63_manvr):
    amp = amp_manvr * (1.0 - exp(-tgo_tgt / tgo63_manvr))
    an_manvr = amp * sin(frq_manvr * tgo_tgt)
    al_manvr = amp * cos(frq_manvr * tgo_tgt)
    return al_manvr, an_manvr


def _plant(
    store,
    *,
    gmax=GMAX,
    grav=GRAV,
    dvta=DVTA,
    utaa=UTAA,
    woea=WOEA,
    flag_exo=1,
    tgo_tgt=TGO_TGT,
):
    store.define(Field("gmax", gmax, "real", "out", "aerodynamics"))
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("dvta", dvta, "real", "out", "sensor", ("com",)))
    store.define(Field("UTAA", utaa, "vec", "out", "sensor"))
    store.define(Field("WOEA", woea, "vec", "out", "sensor"))
    store.define(Field("flag_exo", flag_exo, "int", "data", "control"))
    store.define(Field("tgo_tgt", tgo_tgt, "real", "diag", "sensor"))


def _defined():
    vehicle = _Vehicle()
    guidance = Sam6RocketGuidance()
    guidance.define(vehicle)
    return vehicle, guidance


def _ready(
    *,
    mguide=0,
    gmax=GMAX,
    grav=GRAV,
    gnav=GNAV,
    dvta=DVTA,
    utaa=UTAA,
    woea=WOEA,
    flag_exo=1,
    tgo_tgt=TGO_TGT,
    tgo_manvr=TGO_MANVR,
    amp_manvr=AMP_MANVR,
    frq_manvr=FRQ_MANVR,
    tgo63_manvr=TGO63_MANVR,
    plant=True,
):
    vehicle, guidance = _defined()
    store = vehicle.store
    store.set("mguide", mguide)
    store.set("gnav", gnav)
    store.set("tgo_manvr", tgo_manvr)
    store.set("amp_manvr", amp_manvr)
    store.set("frq_manvr", frq_manvr)
    store.set("tgo63_manvr", tgo63_manvr)
    if plant:
        _plant(
            store,
            gmax=gmax,
            grav=grav,
            dvta=dvta,
            utaa=utaa,
            woea=woea,
            flag_exo=flag_exo,
            tgo_tgt=tgo_tgt,
        )
    return vehicle, guidance


def test_name_is_guidance():
    assert Sam6RocketGuidance.name == "guidance"


def test_define_cpp_fields():
    vehicle, _ = _defined()
    store = vehicle.store
    assert tuple(store.names()) == DEFINED
    for name in DEFINED:
        field = store.field(name)
        assert field.module == "guidance", name
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        if name in INT_FIELDS:
            assert field.type == "int", name
            assert store.get(name) == 0, name
        else:
            assert field.type == "real", name
            assert _approx(store.get(name), 0.0), name
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_mguide_0_gmax_10_writes_zero_commands():
    vehicle, guidance = _ready(mguide=0, gmax=10.0)
    store = vehicle.store
    store.set("ancomx", 9.0)
    store.set("alcomx", -3.0)
    store.set("annx", 4.0)
    store.set("allx", 5.0)
    guidance.execute(vehicle, _ctx())
    assert _approx(store.get("ancomx"), 0.0)
    assert _approx(store.get("alcomx"), 0.0)
    assert _approx(store.get("annx"), 0.0)
    assert _approx(store.get("allx"), 0.0)
    assert _approx(store.get("an_manvr"), 0.0)
    assert _approx(store.get("al_manvr"), 0.0)


def test_mguide_22_manvr_2_raises():
    vehicle, guidance = _ready(mguide=22)
    with pytest.raises(ValueError):
        guidance.execute(vehicle, _ctx())
    assert _approx(vehicle.store.get("ancomx"), 0.0)
    assert _approx(vehicle.store.get("alcomx"), 0.0)


def test_guid_mode_2_raises():
    vehicle, guidance = _ready(mguide=2)
    with pytest.raises(ValueError):
        guidance.execute(vehicle, _ctx())


def test_guid_manvr_2_mode_1_raises():
    vehicle, guidance = _ready(mguide=21)
    with pytest.raises(ValueError):
        guidance.execute(vehicle, _ctx())


def test_guid_mode_1_pronav_unrestricted():
    vehicle, guidance = _ready(mguide=1, gmax=GMAX, woea=WOEA, utaa=UTAA)
    guidance.execute(vehicle, _ctx())
    store = vehicle.store
    allx, annx = _cpp_pronav(WOEA, UTAA, GNAV, DVTA, GRAV)
    alcomx, ancomx = _cpp_limit(allx, annx, GMAX)
    assert _approx(allx, 0.6)
    assert _approx(annx, 0.4)
    assert _approx(store.get("allx"), 0.6)
    assert _approx(store.get("annx"), 0.4)
    assert _approx(store.get("alcomx"), 0.6)
    assert _approx(store.get("ancomx"), 0.4)
    assert _approx(store.get("alcomx"), alcomx)
    assert _approx(store.get("ancomx"), ancomx)
    assert _approx(store.get("an_manvr"), 0.0)
    assert _approx(store.get("al_manvr"), 0.0)


def test_guid_mode_1_pronav_limiter_caps_gmax():
    gmax = 0.5
    vehicle, guidance = _ready(mguide=1, gmax=gmax, woea=WOEA, utaa=UTAA)
    guidance.execute(vehicle, _ctx())
    allx, annx = _cpp_pronav(WOEA, UTAA, GNAV, DVTA, GRAV)
    alcomx, ancomx = _cpp_limit(allx, annx, gmax)
    aax = sqrt(allx * allx + annx * annx)
    assert aax > gmax
    store = vehicle.store
    assert _approx(store.get("allx"), 0.6)
    assert _approx(store.get("annx"), 0.4)
    assert _approx(store.get("alcomx"), alcomx)
    assert _approx(store.get("ancomx"), ancomx)
    mag = sqrt(store.get("alcomx") ** 2 + store.get("ancomx") ** 2)
    assert _approx(mag, gmax)


def test_circular_limiter_or_zeros_normal_when_lateral_small():
    vehicle, guidance = _ready(
        mguide=1, gmax=GMAX, woea=WOEA_LATERAL_ONLY, utaa=UTAA
    )
    guidance.execute(vehicle, _ctx())
    allx, annx = _cpp_pronav(WOEA_LATERAL_ONLY, UTAA, GNAV, DVTA, GRAV)
    alcomx, ancomx = _cpp_limit(allx, annx, GMAX)
    assert _approx(allx, 0.0)
    assert _approx(annx, 1.0)
    store = vehicle.store
    assert _approx(store.get("allx"), 0.0)
    assert _approx(store.get("annx"), 1.0)
    assert _approx(store.get("alcomx"), 1.0)
    assert _approx(store.get("ancomx"), 0.0)
    assert _approx(store.get("alcomx"), alcomx)
    assert _approx(store.get("ancomx"), ancomx)


def test_guid_manvr_1_spiral_as_cpp():
    vehicle, guidance = _ready(
        mguide=10,
        flag_exo=1,
        tgo_tgt=TGO_TGT,
        tgo_manvr=TGO_MANVR,
    )
    guidance.execute(vehicle, _ctx())
    al_manvr, an_manvr = _cpp_spiral(
        TGO_TGT, AMP_MANVR, FRQ_MANVR, TGO63_MANVR
    )
    alcomx, ancomx = _cpp_limit(al_manvr, an_manvr, GMAX)
    store = vehicle.store
    assert _approx(store.get("al_manvr"), al_manvr)
    assert _approx(store.get("an_manvr"), an_manvr)
    assert _approx(store.get("allx"), al_manvr)
    assert _approx(store.get("annx"), an_manvr)
    assert _approx(store.get("alcomx"), alcomx)
    assert _approx(store.get("ancomx"), ancomx)


def test_guid_manvr_1_and_mode_1_adds_spiral_to_pronav():
    vehicle, guidance = _ready(mguide=11, flag_exo=1, tgo_tgt=TGO_TGT)
    guidance.execute(vehicle, _ctx())
    allx, annx = _cpp_pronav(WOEA, UTAA, GNAV, DVTA, GRAV)
    al_manvr, an_manvr = _cpp_spiral(
        TGO_TGT, AMP_MANVR, FRQ_MANVR, TGO63_MANVR
    )
    allx = allx + al_manvr
    annx = annx + an_manvr
    alcomx, ancomx = _cpp_limit(allx, annx, GMAX)
    store = vehicle.store
    assert _approx(store.get("al_manvr"), al_manvr)
    assert _approx(store.get("an_manvr"), an_manvr)
    assert _approx(store.get("allx"), allx)
    assert _approx(store.get("annx"), annx)
    assert _approx(store.get("alcomx"), alcomx)
    assert _approx(store.get("ancomx"), ancomx)


def test_spiral_skipped_when_flag_exo_off():
    vehicle, guidance = _ready(mguide=11, flag_exo=0, tgo_tgt=TGO_TGT)
    guidance.execute(vehicle, _ctx())
    allx, annx = _cpp_pronav(WOEA, UTAA, GNAV, DVTA, GRAV)
    alcomx, ancomx = _cpp_limit(allx, annx, GMAX)
    store = vehicle.store
    assert _approx(store.get("an_manvr"), 0.0)
    assert _approx(store.get("al_manvr"), 0.0)
    assert _approx(store.get("allx"), allx)
    assert _approx(store.get("annx"), annx)
    assert _approx(store.get("alcomx"), alcomx)
    assert _approx(store.get("ancomx"), ancomx)


def test_spiral_skipped_when_tgo_not_below_manvr():
    vehicle, guidance = _ready(
        mguide=10, flag_exo=1, tgo_tgt=TGO_MANVR, tgo_manvr=TGO_MANVR
    )
    guidance.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("an_manvr"), 0.0)
    assert _approx(store.get("al_manvr"), 0.0)
    assert _approx(store.get("ancomx"), 0.0)
    assert _approx(store.get("alcomx"), 0.0)


def test_initialize_and_terminate_are_pass():
    vehicle, guidance = _ready(mguide=0)
    ctx = _ctx()
    assert guidance.initialize(vehicle, ctx) is None
    guidance.execute(vehicle, ctx)
    assert guidance.terminate(vehicle, ctx) is None
    assert _approx(vehicle.store.get("ancomx"), 0.0)


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.flat6.sam6.rocket as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "Plane5" not in src
    assert "Plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src

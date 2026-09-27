from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import DEG, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.sraam6.control import Sraam6Control

RTOL = 1e-12
ATOL = 1e-14

PHICOMX = 0.0
WRCL = 20.0
ZRCL = 0.9
DLP = -2.0
DLD = 1.0
PHIBLX = 0.0
PP = 0.0

DEFINED = (
    "maut",
    "mfreeze",
    "wacl",
    "zacl",
    "pacl",
    "alimit",
    "dqlimx",
    "drlimx",
    "dplimx",
    "phicomx",
    "wrcl",
    "zrcl",
    "yyd",
    "yy",
    "zzd",
    "zz",
    "dpcx",
    "dqcx",
    "drcx",
    "GAINFB",
    "gainp",
    "gkp",
    "gkphi",
    "fspb2m",
    "fspb2mt",
    "fspb3m",
    "fspb3mt",
    "qqxm",
    "qqxmt",
    "rrxm",
    "rrxmt",
    "dqcxm",
    "dqcxmt",
    "drcxm",
    "drcxmt",
    "isetc2",
    "factwacl",
    "factzacl",
    "zetlagr",
    "ratelimx",
    "zrate",
    "grate",
    "wnlagr",
)
INT_FIELDS = ("maut", "mfreeze")
VEC_FIELDS = ("GAINFB",)
STATE_FIELDS = ("yyd", "yy", "zzd", "zz")
OUT_FIELDS = ("dpcx", "dqcx", "drcx")
DATA_REALS = (
    "wacl",
    "zacl",
    "pacl",
    "alimit",
    "dqlimx",
    "drlimx",
    "dplimx",
    "phicomx",
    "wrcl",
    "zrcl",
    "gainp",
    "factwacl",
    "factzacl",
    "zetlagr",
    "ratelimx",
)
DIAG_REAL = (
    "gkp",
    "gkphi",
    "fspb2m",
    "fspb2mt",
    "fspb3m",
    "fspb3mt",
    "qqxm",
    "qqxmt",
    "rrxm",
    "rrxmt",
    "dqcxm",
    "dqcxmt",
    "drcxm",
    "drcxmt",
    "zrate",
    "grate",
    "wnlagr",
)
EXTERNALS = (
    "phiblx",
    "pp",
    "dlp",
    "dld",
    "time",
    "WBECB",
    "phiblcx",
    "phibdcx",
    "ppcx",
    "ppx",
    "dllp",
    "dllda",
    "delacx",
    "delecx",
    "delrcx",
    "dna",
    "dnd",
    "dma",
    "dmq",
    "dmd",
    "dvbe",
    "qq",
    "rr",
)


def _ctx():
    return SimContext(
        sim_time=0.0,
        int_step=0.001,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _externals(store, *, phiblx=PHIBLX, pp=PP, dlp=DLP, dld=DLD):
    store.define(Field("phiblx", phiblx, "real", "diag", "kinematics"))
    store.define(Field("pp", pp, "real", "state", "euler"))
    store.define(Field("dlp", dlp, "real", "out", "aerodynamics"))
    store.define(Field("dld", dld, "real", "out", "aerodynamics"))


def _ready(**kw):
    vehicle = SimpleNamespace(store=StateStore())
    ctrl = Sraam6Control()
    ctrl.define(vehicle)
    _externals(vehicle.store, **kw)
    store = vehicle.store
    store.set("phicomx", PHICOMX)
    store.set("wrcl", WRCL)
    store.set("zrcl", ZRCL)
    ctrl.initialize(vehicle, _ctx())
    return vehicle, ctrl


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _control_roll(store):
    phicomx = store.get("phicomx")
    wrcl = store.get("wrcl")
    zrcl = store.get("zrcl")
    phiblx = store.get("phiblx")
    dlp = store.get("dlp")
    dld = store.get("dld")
    pp = store.get("pp")
    gkp = (2.0 * zrcl * wrcl + dlp) / dld
    gkphi = wrcl * wrcl / dld
    ephi = gkphi * (phicomx - phiblx) * RAD
    dpc = ephi - gkp * pp
    dpcx = dpc * DEG
    return dpcx, gkp, gkphi


def test_name_is_control():
    assert Sraam6Control.name == "control"
    assert Sraam6Control().name == "control"


def test_define_registers_cpp_fields_not_externals():
    vehicle = SimpleNamespace(store=StateStore())
    Sraam6Control().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    for name in DEFINED:
        assert store.field(name).module == "control"
    for name in INT_FIELDS:
        assert store.get(name) == 0
        assert store.field(name).type == "int"
        assert store.field(name).role == "data"
        assert store.field(name).outputs == ()
    zeros = np.zeros(3)
    for name in VEC_FIELDS:
        np.testing.assert_array_equal(store.get(name), zeros)
        assert store.get(name).shape == (3,)
        assert store.field(name).type == "vec"
        assert store.field(name).role == "diag"
        assert store.field(name).outputs == ()
    for name in STATE_FIELDS:
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "state"
        assert store.field(name).outputs == ()
    for name in OUT_FIELDS:
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "out"
        assert store.field(name).outputs == ()
    for name in DATA_REALS:
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "data"
        assert store.field(name).outputs == ()
    for name in DIAG_REAL:
        assert store.get(name) == 0.0
        assert store.field(name).type == "real"
        assert store.field(name).role == "diag"
        assert store.field(name).outputs == ()
    assert store.field("isetc2").type == "real"
    assert store.field("isetc2").role == "init"
    assert store.field("isetc2").outputs == ()
    assert store.get("isetc2") == 0.0
    for name in EXTERNALS:
        assert name not in store.names()


def test_initialize_is_pass():
    vehicle, _ctrl = _ready()
    store = vehicle.store
    assert store.get("dpcx") == 0.0
    assert store.get("dqcx") == 0.0
    assert store.get("drcx") == 0.0
    assert store.get("gkp") == 0.0
    assert store.get("gkphi") == 0.0


def test_execute_remains_pass():
    vehicle, ctrl = _ready(phiblx=10.0, pp=0.0)
    store = vehicle.store
    store.set("dpcx", 7.0)
    store.set("gkp", 1.5)
    store.set("gkphi", 2.5)
    assert ctrl.execute(vehicle, _ctx()) is None
    assert store.get("dpcx") == 7.0
    assert store.get("gkp") == 1.5
    assert store.get("gkphi") == 2.5
    assert store.get("dqcx") == 0.0
    assert store.get("drcx") == 0.0


def test_control_roll_zero_attitude_matches_cpp():
    vehicle, ctrl = _ready(phiblx=0.0, pp=0.0, dlp=-2.0, dld=1.0)
    want, gkp, gkphi = _control_roll(vehicle.store)
    assert ctrl.control_roll(vehicle) is None
    store = vehicle.store
    assert _approx(store.get("dpcx"), want)
    assert _approx(store.get("gkp"), gkp)
    assert _approx(store.get("gkphi"), gkphi)
    assert store.get("dpcx") == 0.0
    assert gkp != 0.0
    assert gkphi != 0.0
    assert store.get("dqcx") == 0.0
    assert store.get("drcx") == 0.0


def test_control_roll_phiblx_10_nonzero_command():
    vehicle, ctrl = _ready(phiblx=10.0, pp=0.0, dlp=-2.0, dld=1.0)
    want, gkp, gkphi = _control_roll(vehicle.store)
    ctrl.control_roll(vehicle)
    store = vehicle.store
    assert _approx(store.get("dpcx"), want)
    assert store.get("dpcx") != 0.0
    assert _approx(store.get("gkp"), gkp)
    assert _approx(store.get("gkphi"), gkphi)


def test_control_roll_reads_phiblx_pp_not_ins():
    vehicle, ctrl = _ready(phiblx=10.0, pp=0.05, dlp=-2.0, dld=1.0)
    vehicle.store.define(Field("phiblcx", 45.0, "real", "out", "ins"))
    vehicle.store.define(Field("WBECB", (0.9, 0.0, 0.0), "vec", "out", "ins"))
    vehicle.store.define(Field("ppx", 99.0, "real", "out", "euler"))
    want, gkp, gkphi = _control_roll(vehicle.store)
    ctrl.control_roll(vehicle)
    assert _approx(vehicle.store.get("dpcx"), want)
    assert _approx(vehicle.store.get("gkp"), gkp)
    assert _approx(vehicle.store.get("gkphi"), gkphi)
    decoy_gkp = (2.0 * ZRCL * WRCL + DLP) / DLD
    ephi = (WRCL * WRCL / DLD) * (PHICOMX - 45.0) * RAD
    dpc = ephi - decoy_gkp * 0.9
    assert vehicle.store.get("dpcx") != pytest.approx(dpc * DEG, rel=RTOL, abs=ATOL)


def test_control_roll_does_not_scale_pp_by_rad():
    vehicle, ctrl = _ready(phiblx=10.0, pp=0.05, dlp=-2.0, dld=1.0)
    want, gkp, gkphi = _control_roll(vehicle.store)
    ctrl.control_roll(vehicle)
    ephi = gkphi * (PHICOMX - 10.0) * RAD
    wrong = (ephi - gkp * 0.05 * RAD) * DEG
    assert _approx(vehicle.store.get("dpcx"), want)
    assert vehicle.store.get("dpcx") != pytest.approx(wrong, rel=RTOL, abs=ATOL)


def test_terminate_exists_and_is_pass():
    vehicle, ctrl = _ready()
    assert ctrl.terminate(vehicle, _ctx()) is None
    assert vehicle.store.get("dpcx") == 0.0
    assert vehicle.store.get("gkp") == 0.0

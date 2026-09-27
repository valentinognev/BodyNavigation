from types import SimpleNamespace

import numpy as np
import pytest

from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.round6.hyper6.guidance import Hyper6Guidance

PLOT = ("plot",)

# C++ Hyper::def_guidance order (unused slots 401, 433, 447-449 omitted).
FIELDS = {
    "mguide": ("int", "data", 0, ()),
    "line_gain": ("real", "data", 0.0, ()),
    "nl_gain_fact": ("real", "data", 0.0, ()),
    "decrement": ("real", "data", 0.0, ()),
    "wp_lonx": ("real", "data", 0.0, ()),
    "wp_latx": ("real", "data", 0.0, ()),
    "wp_alt": ("real", "data", 0.0, ()),
    "psifdx": ("real", "data", 0.0, ()),
    "thtfdx": ("real", "data", 0.0, ()),
    "point_gain": ("real", "data", 0.0, ()),
    "wp_sltrange": ("real", "diag", 999999.0, ()),
    "nl_gain": ("real", "diag", 0.0, ()),
    "VBEO": ("vec", "diag", (0.0, 0.0, 0.0), ()),
    "VBEF": ("vec", "diag", (0.0, 0.0, 0.0), ()),
    "wp_grdrange": ("real", "diag", 999999.0, ()),
    "SWBD": ("vec", "out", (0.0, 0.0, 0.0), ()),
    "rad_min": ("real", "diag", 0.0, ()),
    "rad_geometric": ("real", "diag", 0.0, ()),
    "wp_flag": ("int", "diag", 0, ()),
    "gnav": ("real", "data", 0.0, ()),
    "aycomx": ("real", "out", 0.0, PLOT),
    "azcomx": ("real", "out", 0.0, PLOT),
    "tgoc": ("real", "diag", 0.0, PLOT),
    "dtbc": ("real", "diag", 0.0, ()),
    "psiobcx": ("real", "diag", 0.0, ()),
    "thtobcx": ("real", "diag", 0.0, ()),
    "SBTHC": ("vec", "diag", (0.0, 0.0, 0.0), PLOT),
    "init_flag": ("int", "init", 1, ()),
    "time_ltg": ("real", "diag", 0.0, ()),
    "UTBC": ("vec", "out", (0.0, 0.0, 0.0), PLOT),
    "RBIAS": ("vec", "save", (0.0, 0.0, 0.0), ()),
    "beco_flag": ("int", "diag", 0, ()),
    "inisw_flag": ("int", "init", 1, ()),
    "skip_flag": ("int", "init", 1, ()),
    "ipas_flag": ("int", "init", 1, ()),
    "ipas2_flag": ("int", "init", 1, ()),
    "print_flag": ("int", "init", 1, ()),
    "ltg_count": ("int", "save", 0, ()),
    "ltg_step": ("real", "data", 0.0, ()),
    "dbi_desired": ("real", "data", 0.0, ()),
    "dvbi_desired": ("real", "data", 0.0, ()),
    "thtvdx_desired": ("real", "data", 0.0, ()),
    "num_stages": ("int", "data", 0, ()),
    "delay_ignition": ("real", "data", 0.0, ()),
    "amin": ("real", "data", 0.0, ()),
    "char_time1": ("real", "data", 0.0, ()),
    "char_time2": ("real", "data", 0.0, ()),
    "char_time3": ("real", "data", 0.0, ()),
    "exhaust_vel1": ("real", "data", 0.0, ()),
    "exhaust_vel2": ("real", "data", 0.0, ()),
    "exhaust_vel3": ("real", "data", 0.0, ()),
    "burnout_epoch1": ("real", "data", 0.0, ()),
    "burnout_epoch2": ("real", "data", 0.0, ()),
    "burnout_epoch3": ("real", "data", 0.0, ()),
    "lamd_limit": ("real", "data", 0.0, ()),
    "RGRAV": ("vec", "save", (0.0, 0.0, 0.0), ()),
    "RGO": ("vec", "save", (0.0, 0.0, 0.0), ()),
    "VGO": ("vec", "save", (0.0, 0.0, 0.0), ()),
    "SDII": ("vec", "save", (0.0, 0.0, 0.0), ()),
    "UD": ("vec", "save", (0.0, 0.0, 0.0), ()),
    "UY": ("vec", "save", (0.0, 0.0, 0.0), ()),
    "UZ": ("vec", "save", (0.0, 0.0, 0.0), ()),
    "vgom": ("real", "diag", 0.0, ()),
    "tgo": ("real", "save", 0.0, ()),
    "nst": ("int", "save", 0, ()),
    "ULAM": ("vec", "diag", (0.0, 0.0, 0.0), ()),
    "LAMD": ("vec", "diag", (0.0, 0.0, 0.0), ()),
    "isp_fuel": ("real", "out", 0.0, ()),
    "burntime": ("real", "out", 0.0, ()),
    "nstmax": ("int", "diag", 0, ()),
    "lamd": ("real", "diag", 0.0, PLOT),
    "dpd": ("real", "diag", 0.0, ()),
    "dbd": ("real", "diag", 0.0, ()),
    "gnavpn": ("real", "data", 0.0, ()),
    "gnavps": ("real", "data", 0.0, ()),
    "gs_flag": ("int", "init", 1, ()),
    "time_gs": ("real", "data", 0.0, ()),
    "num_burns": ("int", "data", 0, ()),
    "closing_rate": ("real", "data", 0.0, ()),
    "orbital_rate": ("real", "data", 0.0, ()),
    "satl1": ("real", "data", 0.0, ()),
    "satl2": ("real", "data", 0.0, ()),
    "satl3": ("real", "data", 0.0, ()),
    "dtime_gs": ("real", "save", 0.0, ()),
    "length_gs": ("real", "save", 0.0, ()),
    "para_gs": ("real", "save", 0.0, ()),
    "UB0AL": ("vec", "save", (0.0, 0.0, 0.0), ()),
    "epoch_gs": ("real", "save", 0.0, ()),
    "counter_gs": ("int", "save", 1, ()),
    "VBTLM": ("vec", "save", (0.0, 0.0, 0.0), ()),
    "DELTA_V": ("vec", "save", (0.0, 0.0, 0.0), ()),
    "burn_flag": ("int", "save", 1, ()),
    "SBTL": ("vec", "diag", (0.0, 0.0, 0.0), PLOT),
    "VBTL": ("vec", "diag", (0.0, 0.0, 0.0), ()),
    "delta_v": ("real", "diag", 0.0, ()),
}
DEFINED = tuple(FIELDS)

EXTERNALS = (
    "time",
    "grav",
    "maut",
    "mprop",
    "mseek",
    "STBIK",
    "VTBIK",
    "TBIC",
    "SBIIC",
    "VBIIC",
    "FSPCB",
    "alcomx",
    "ancomx",
    "phicomx",
    "philimx",
    "lonx",
    "latx",
    "alt",
    "SBII",
    "VBII",
    "dbi",
    "dvbi",
    "dvbe",
)
CONTROL_OWNED = ("alcomx", "ancomx", "phicomx")
NO_HELPERS = (
    "guidance_ltg",
    "guidance_line",
    "guidance_pronav",
    "guidance_arc",
    "guidance_AGL",
    "guidance_glideslope",
)
GUIDANCE_SENTINELS = (
    "mguide",
    "wp_flag",
    "aycomx",
    "azcomx",
    "init_flag",
    "UTBC",
    "VBEO",
    "wp_sltrange",
)


def _ctx(int_step=0.01):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _ready(mguide=0):
    vehicle = SimpleNamespace(store=StateStore())
    guid = Hyper6Guidance()
    guid.define(vehicle)
    vehicle.store.set("mguide", mguide)
    guid.initialize(vehicle, _ctx())
    return vehicle, guid


def test_name_is_guidance():
    assert Hyper6Guidance().name == "guidance"


def test_define_registers_cpp_def_guidance_fields():
    vehicle = SimpleNamespace(store=StateStore())
    Hyper6Guidance().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    zeros = np.zeros(3)
    for name, (ftype, role, default, outputs) in FIELDS.items():
        field = store.field(name)
        assert field.module == "guidance"
        assert field.type == ftype
        assert field.role == role
        assert field.outputs == outputs
        if ftype == "int":
            assert store.get(name) == default
            assert type(store.get(name)) is int
        elif ftype == "real":
            assert store.get(name) == default
        else:
            np.testing.assert_array_equal(store.get(name), zeros)
            assert store.get(name).shape == (3,)
    for name in EXTERNALS:
        assert name not in store.names()


def test_define_does_not_register_newton_or_control_names():
    vehicle = SimpleNamespace(store=StateStore())
    Hyper6Guidance().define(vehicle)
    store = vehicle.store
    for name in EXTERNALS:
        with pytest.raises(KeyError):
            store.get(name)


def test_initialize_is_pass():
    vehicle, _guid = _ready()
    store = vehicle.store
    assert store.get("mguide") == 0
    assert store.get("line_gain") == 0.0
    assert store.get("wp_sltrange") == 999999.0
    assert store.get("wp_grdrange") == 999999.0
    assert store.get("init_flag") == 1
    assert store.get("gs_flag") == 1
    assert store.get("counter_gs") == 1
    assert store.get("burn_flag") == 1
    assert store.get("wp_flag") == 0
    np.testing.assert_array_equal(store.get("VBEO"), np.zeros(3))
    np.testing.assert_array_equal(store.get("UTBC"), np.zeros(3))


def test_terminate_exists_and_is_pass():
    vehicle, guid = _ready()
    store = vehicle.store
    store.set("mguide", 0)
    store.set("wp_flag", 1)
    store.set("aycomx", 3.0)
    sentinel = np.array([9.0, 8.0, 7.0])
    store.set("UTBC", sentinel)
    guid.terminate(vehicle, _ctx())
    assert store.get("mguide") == 0
    assert store.get("wp_flag") == 1
    assert store.get("aycomx") == 3.0
    np.testing.assert_array_equal(store.get("UTBC"), sentinel)


def test_no_ltg_line_pronav_helpers():
    guid = Hyper6Guidance()
    for name in NO_HELPERS:
        assert not hasattr(guid, name)


def test_execute_mguide_zero_does_not_raise():
    vehicle, guid = _ready(mguide=0)
    guid.execute(vehicle, _ctx())


def test_execute_mguide_zero_does_not_write_control_or_diagnostics():
    vehicle, guid = _ready(mguide=0)
    store = vehicle.store
    for name in CONTROL_OWNED:
        store.define(Field(name, 7.0, "real", "data", "control"))
    store.set("wp_flag", 1)
    store.set("aycomx", 11.0)
    store.set("azcomx", 12.0)
    store.set("init_flag", 0)
    store.set("wp_sltrange", 42.0)
    sentinel = np.array([9.0, 8.0, 7.0])
    store.set("UTBC", sentinel)
    store.set("VBEO", sentinel)
    guid.execute(vehicle, _ctx())
    for name in CONTROL_OWNED:
        assert store.get(name) == 7.0
    assert store.get("mguide") == 0
    assert store.get("wp_flag") == 1
    assert store.get("aycomx") == 11.0
    assert store.get("azcomx") == 12.0
    assert store.get("init_flag") == 0
    assert store.get("wp_sltrange") == 42.0
    np.testing.assert_array_equal(store.get("UTBC"), sentinel)
    np.testing.assert_array_equal(store.get("VBEO"), sentinel)


def test_execute_mguide_zero_does_not_require_newton_or_control_names():
    vehicle, guid = _ready(mguide=0)
    store = vehicle.store
    for name in EXTERNALS:
        assert name not in store.names()
    guid.execute(vehicle, _ctx())
    for name in EXTERNALS:
        assert name not in store.names()


def test_execute_mguide_five_raises():
    vehicle, guid = _ready(mguide=5)
    store = vehicle.store
    for name in CONTROL_OWNED:
        store.define(Field(name, 7.0, "real", "data", "control"))
    store.set("wp_flag", 1)
    sentinel = np.array([9.0, 8.0, 7.0])
    store.set("UTBC", sentinel)
    with pytest.raises(ValueError, match="unknown mguide"):
        guid.execute(vehicle, _ctx())
    for name in CONTROL_OWNED:
        assert store.get(name) == 7.0
    for name in EXTERNALS:
        if name not in CONTROL_OWNED:
            assert name not in store.names()
    assert store.get("wp_flag") == 1
    np.testing.assert_array_equal(store.get("UTBC"), sentinel)


@pytest.mark.parametrize("mguide", (30, 33, 3, 4, 6, 7, 8, -1, 99))
def test_execute_unknown_mguide_raises(mguide):
    vehicle, guid = _ready(mguide=mguide)
    with pytest.raises(ValueError, match="unknown mguide"):
        guid.execute(vehicle, _ctx())
    for name in GUIDANCE_SENTINELS:
        if name == "mguide":
            assert vehicle.store.get(name) == mguide
        else:
            ftype, _role, default, _outputs = FIELDS[name]
            if ftype == "vec":
                np.testing.assert_array_equal(vehicle.store.get(name), np.zeros(3))
            else:
                assert vehicle.store.get(name) == default

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import R
from cadac.env.gravity import gravity
from cadac.env.us76 import atmosphere76
from cadac.io.asc_deck import parse_asc_deck
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore
from cadac.tables.lookup import Datadeck
from cadac.vehicles.flat6.agm6.environment import Agm6Environment

RTOL = 1e-12
ATOL = 1e-14
PLOT = ("plot",)
SCRN_PLOT = ("scrn", "plot")
SCRN_PLOT_COM = ("scrn", "plot", "com")
ZEROS3 = (0.0, 0.0, 0.0)

HBE = 7000.0
VBEL = np.array([293.0, 0.0, 0.0], dtype=float)
DVBE = 293.0
MAIR_WEATHER = 212
AGM6 = Path(__file__).resolve().parents[3] / "CADAC_Simulations/AGM6_250217/AGM6"
WEATHER_ASC = AGM6 / "weather_deck.asc"

# C++ Flat6::def_environment order, plus VBAL for reused Flat6Kinematics.
FIELDS = {
    "mair": ("int", "data", 0, ()),
    "warning_flag": ("int", "init", 0, ()),
    "press": ("real", "out", 0.0, ()),
    "rho": ("real", "out", 0.0, ()),
    "vsound": ("real", "diag", 0.0, ()),
    "grav": ("real", "out", 0.0, ()),
    "vmach": ("real", "out", 0.0, SCRN_PLOT_COM),
    "pdynmc": ("real", "out", 0.0, SCRN_PLOT),
    "tempk": ("real", "out", 0.0, ()),
    "mfreeze_evrn": ("int", "save", 0, ()),
    "pdynmcf": ("real", "save", 0.0, ()),
    "vmachf": ("real", "save", 0.0, ()),
    "GRAVL": ("vec", "out", ZEROS3, ()),
    "dvae": ("real", "data", 0.0, ()),
    "dvael": ("real", "data", 0.0, ()),
    "waltl": ("real", "data", 0.0, ()),
    "dvaeh": ("real", "data", 0.0, ()),
    "walth": ("real", "data", 0.0, ()),
    "vaed3": ("real", "data", 0.0, ()),
    "psiwdx": ("real", "data", 0.0, ()),
    "twind": ("real", "data", 0.1, ()),
    "VAELS": ("vec", "state", ZEROS3, ()),
    "VAELSD": ("vec", "state", ZEROS3, ()),
    "VAEL": ("vec", "out", ZEROS3, PLOT),
    "dvba": ("real", "out", 0.0, ()),
    "markov_value": ("real", "save", 0.0, ()),
    "turb_length": ("real", "data", 0.0, ()),
    "turb_sigma": ("real", "data", 0.0, ()),
    "taux1": ("real", "state", 0.0, ()),
    "taux1d": ("real", "state", 0.0, ()),
    "taux2": ("real", "state", 0.0, ()),
    "taux2d": ("real", "state", 0.0, ()),
    "tau": ("real", "diag", 0.0, ()),
    "gauss_value": ("real", "diag", 0.0, ()),
    "tempc": ("real", "diag", 0.0, ()),
    "VBAL": ("vec", "out", ZEROS3, ()),
}
DEFINED = tuple(FIELDS)
INT_FIELDS = ("mair", "warning_flag", "mfreeze_evrn")
VEC_FIELDS = ("GRAVL", "VAELS", "VAELSD", "VAEL", "VBAL")
NEWTON = ("hbe", "VBEL", "dvbe")
SKIP_IF_ABSENT = ("trcond", "mfreeze")


def _ctx(int_step=0.001):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _plant_newton(store, *, hbe=HBE, vbel=VBEL, dvbe=DVBE):
    store.define(Field("hbe", hbe, "real", "out", "newton"))
    store.define(Field("VBEL", tuple(vbel), "vec", "state", "newton"))
    store.define(Field("dvbe", dvbe, "real", "out", "newton"))


def _weather_deck():
    _, tables = parse_asc_deck(WEATHER_ASC)
    return Datadeck.from_tables(tables)


def _plant_kinematics(store):
    store.define(Field("TBD", np.eye(3), "mat", "out", "kinematics"))
    store.define(Field("alppx", 0.0, "real", "out", "kinematics"))
    store.define(Field("phipx", 0.0, "real", "out", "kinematics"))


def _ready(*, mair=0, hbe=HBE, vbel=VBEL, dvbe=DVBE, weather_deck=None, kinematics=False):
    vehicle = SimpleNamespace(store=StateStore())
    env = Agm6Environment(weather_deck=weather_deck)
    env.define(vehicle)
    _plant_newton(vehicle.store, hbe=hbe, vbel=vbel, dvbe=dvbe)
    if kinematics:
        _plant_kinematics(vehicle.store)
    vehicle.store.set("mair", mair)
    env.initialize(vehicle, _ctx())
    return vehicle, env


def test_name_is_environment():
    assert Agm6Environment.name == "environment"
    assert Agm6Environment().name == "environment"


def test_constructor_weather_deck_none():
    sig = inspect.signature(Agm6Environment.__init__)
    assert list(sig.parameters) == ["self", "weather_deck"]
    assert sig.parameters["weather_deck"].default is None
    env = Agm6Environment()
    assert env.weather_deck is None
    env = Agm6Environment(weather_deck=None)
    assert env.weather_deck is None


def test_define_registers_cpp_fields_plus_vbal():
    vehicle = SimpleNamespace(store=StateStore())
    Agm6Environment().define(vehicle)
    store = vehicle.store
    assert list(store.names()) == list(DEFINED)
    zeros3 = np.zeros(3)
    for name, (ftype, role, default, outputs) in FIELDS.items():
        field = store.field(name)
        assert field.type == ftype, name
        assert field.role == role, name
        assert field.module == "environment"
        assert field.outputs == outputs, name
        if ftype == "int":
            assert store.get(name) == default
        elif ftype == "vec":
            np.testing.assert_allclose(store.get(name), zeros3)
        else:
            assert store.get(name) == pytest.approx(default, rel=RTOL, abs=ATOL)
    for name in NEWTON + SKIP_IF_ABSENT:
        assert name not in store.names()


def test_initialize_sets_dvba_from_dvbe():
    vehicle = SimpleNamespace(store=StateStore())
    env = Agm6Environment()
    env.define(vehicle)
    store = vehicle.store
    store.define(Field("dvbe", DVBE, "real", "out", "newton"))
    env.initialize(vehicle, _ctx())
    assert store.get("dvba") == pytest.approx(DVBE, rel=RTOL, abs=ATOL)


def test_mair_0_us76_vbal_equals_vbel():
    vehicle, env = _ready(mair=0, hbe=HBE, vbel=VBEL)
    env.execute(vehicle, _ctx())
    store = vehicle.store
    rho, press, tempk = atmosphere76(HBE)
    vsound = (1.4 * R * tempk) ** 0.5
    dvba = float(np.linalg.norm(VBEL))
    np.testing.assert_allclose(store.get("rho"), rho, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("press"), press, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("tempk"), tempk, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("grav"), gravity(HBE), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VAEL"), np.zeros(3), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VBAL"), store.get("VBEL"), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("VBAL"), VBEL, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("dvba"), dvba, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(store.get("vsound"), vsound, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        store.get("vmach"), abs(dvba / vsound), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        store.get("pdynmc"), 0.5 * rho * dvba * dvba, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        store.get("tempc"), tempk - 273.16, rtol=RTOL, atol=ATOL
    )
    assert np.isfinite(store.get("vmach"))
    assert store.get("mair") == 0


def test_mair_212_no_longer_raises_with_weather_deck():
    vehicle, env = _ready(
        mair=MAIR_WEATHER,
        weather_deck=_weather_deck(),
        kinematics=True,
    )
    vehicle.store.set("turb_length", 100.0)
    vehicle.store.set("turb_sigma", 0.5)
    vehicle.store.set("gauss_value", 0.0)
    env.execute(vehicle, _ctx())
    vael = vehicle.store.get("VAEL")
    assert np.all(np.isfinite(vael))
    np.testing.assert_allclose(
        vehicle.store.get("VBAL"), vehicle.store.get("VBEL") - vael, rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("mair", [100, 20, 3, 300])
def test_unknown_mair_digits_raise(mair):
    vehicle, env = _ready(mair=mair, weather_deck=_weather_deck(), kinematics=True)
    with pytest.raises(ValueError):
        env.execute(vehicle, _ctx())


def test_execute_skips_trcond_mfreeze_when_absent():
    vehicle, env = _ready(mair=0)
    env.execute(vehicle, _ctx())
    store = vehicle.store
    for name in SKIP_IF_ABSENT:
        assert name not in store.names()
    assert np.isfinite(store.get("vmach"))
    assert np.isfinite(store.get("pdynmc"))

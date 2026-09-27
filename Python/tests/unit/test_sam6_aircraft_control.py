from math import atan2, sqrt
from pathlib import Path

import numpy as np
import pytest

from cadac.constants import DEG, EPS, RAD
from cadac.kernel.executive import SimContext
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.vehicles.flat6.sam6.aircraft import Sam6AircraftControl

RTOL = 1e-12
ATOL = 1e-14
GRAV = 9.8
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
YAW90 = ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))

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
ROLES = {
    "phiav": "state",
    "phiavd": "state",
    "tphi": "data",
    "philimx": "data",
    "phiavx": "out",
    "phiavcx": "diag",
    "anx": "state",
    "anxd": "state",
    "tanx": "data",
    "alplimx": "data",
    "ancomx": "diag",
    "clalpha": "data",
    "wingloading": "data",
    "phiavout": "out",
}
OUTPUTS = {
    "phiav": (),
    "phiavd": (),
    "tphi": (),
    "philimx": (),
    "phiavx": ("com",),
    "phiavcx": (),
    "anx": ("com",),
    "anxd": (),
    "tanx": (),
    "alplimx": (),
    "ancomx": (),
    "clalpha": (),
    "wingloading": (),
    "phiavout": (),
}
NOT_DEFINED = (
    "acft_option",
    "ACOML",
    "grav",
    "pdynmc",
    "TVL",
    "time",
    "SAEL",
    "VAEL",
    "FSPA",
    "acc_longx",
    "guid_gain",
    "gturn",
)


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def _approx(got, want):
    return got == pytest.approx(want, rel=RTOL, abs=ATOL)


def _ctx(int_step=0.001):
    return SimContext(
        sim_time=0.0,
        int_step=int_step,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )


def _plant(
    store,
    *,
    grav=GRAV,
    pdynmc=0.0,
    tvl=IDENTITY,
    acft_option=0,
    acoml=(0.0, 0.0, -GRAV),
):
    store.define(Field("grav", grav, "real", "out", "environment"))
    store.define(Field("pdynmc", pdynmc, "real", "out", "environment"))
    store.define(Field("TVL", tvl, "mat", "out", "newton"))
    store.define(Field("acft_option", acft_option, "int", "data", "guidance"))
    store.define(Field("ACOML", acoml, "vec", "out", "guidance"))


def _ready(
    *,
    tphi=0.0,
    philimx=90.0,
    tanx=0.0,
    alplimx=0.0,
    clalpha=0.0,
    wingloading=1.0,
    grav=GRAV,
    pdynmc=0.0,
    tvl=IDENTITY,
    acft_option=0,
    acoml=(0.0, 0.0, -GRAV),
    phiav=0.0,
    phiavd=0.0,
    anx=0.0,
    anxd=0.0,
):
    vehicle = _Vehicle()
    control = Sam6AircraftControl()
    control.define(vehicle)
    control.initialize(vehicle, _ctx())
    store = vehicle.store
    store.set("tphi", tphi)
    store.set("philimx", philimx)
    store.set("tanx", tanx)
    store.set("alplimx", alplimx)
    store.set("clalpha", clalpha)
    store.set("wingloading", wingloading)
    store.set("phiav", phiav)
    store.set("phiavd", phiavd)
    store.set("anx", anx)
    store.set("anxd", anxd)
    _plant(
        store,
        grav=grav,
        pdynmc=pdynmc,
        tvl=tvl,
        acft_option=acft_option,
        acoml=acoml,
    )
    return vehicle, control


def _expected(store, int_step):
    tphi = store.get("tphi")
    philimx = store.get("philimx")
    tanx = store.get("tanx")
    alplimx = store.get("alplimx")
    clalpha = store.get("clalpha")
    wingloading = store.get("wingloading")
    grav = store.get("grav")
    pdynmc = store.get("pdynmc")
    tvl = np.asarray(store.get("TVL"), dtype=float)
    acft_option = store.get("acft_option")
    acoml = np.asarray(store.get("ACOML"), dtype=float)
    phiav = store.get("phiav")
    phiavd = store.get("phiavd")
    anx = store.get("anx")
    anxd = store.get("anxd")
    acomv = tvl @ acoml
    acoma2 = float(acomv[1])
    acoma3 = float(acomv[2])
    if abs(acoma2) < EPS and abs(acoma3) < EPS:
        phiavc = 0.0
    else:
        phiavc = atan2(acoma2, -acoma3)
    phiavcx = phiavc * DEG
    if tphi:
        phiavd_new = (phiavc - phiav) / tphi
        phiav = integrate(phiavd_new, phiavd, phiav, int_step)
        phiavd = phiavd_new
    else:
        phiav = phiavc
    phiavx = phiav * DEG
    if abs(phiavx) >= philimx:
        phiavx = philimx * _sign(phiavx)
    phiavout = phiavx * RAD
    ancomx = sqrt(acoma2 * acoma2 + acoma3 * acoma3) / grav
    if tanx:
        anxd_new = (ancomx - anx) / tanx
        anx = integrate(anxd_new, anxd, anx, int_step)
        anxd = anxd_new
    else:
        anx = ancomx
    if acft_option > 0:
        anlimx = pdynmc * clalpha * alplimx / wingloading
        if abs(anx) >= anlimx:
            anx = anlimx * _sign(anx)
    return {
        "phiav": phiav,
        "phiavd": phiavd,
        "phiavx": phiavx,
        "phiavcx": phiavcx,
        "phiavout": phiavout,
        "ancomx": ancomx,
        "anx": anx,
        "anxd": anxd,
    }


def test_name_is_control():
    assert Sam6AircraftControl().name == "control"


def test_define_registers_cpp_fields():
    vehicle = _Vehicle()
    Sam6AircraftControl().define(vehicle)
    store = vehicle.store
    assert tuple(store.names()) == DEFINED
    for name in DEFINED:
        field = store.field(name)
        assert field.module == "control", name
        assert field.role == ROLES[name], name
        assert field.outputs == OUTPUTS[name], name
        assert field.type == "real", name
        assert _approx(store.get(name), 0.0), name
    for name in NOT_DEFINED:
        assert name not in store.names()


def test_initialize_is_pass():
    vehicle = _Vehicle()
    control = Sam6AircraftControl()
    control.define(vehicle)
    vehicle.store.set("phiavout", 1.0)
    vehicle.store.set("anx", 2.0)
    before_phi = vehicle.store.get("phiavout")
    before_anx = vehicle.store.get("anx")
    assert control.initialize(vehicle, _ctx()) is None
    assert _approx(vehicle.store.get("phiavout"), before_phi)
    assert _approx(vehicle.store.get("anx"), before_anx)


def test_level_flight_phiavout_zero_anx_one():
    vehicle, control = _ready(
        acoml=(0.0, 0.0, -GRAV),
        tvl=IDENTITY,
        tphi=0.0,
        tanx=0.0,
        philimx=90.0,
        acft_option=0,
        grav=GRAV,
    )
    control.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("phiavout"), 0.0)
    assert _approx(store.get("anx"), 1.0)
    assert _approx(store.get("phiavx"), 0.0)
    assert _approx(store.get("ancomx"), 1.0)


def test_bank_from_tvl_times_acoml_not_transpose():
    vehicle, control = _ready(
        acoml=(GRAV, 0.0, 0.0),
        tvl=YAW90,
        tphi=0.0,
        tanx=0.0,
        philimx=90.0,
        acft_option=0,
        grav=GRAV,
    )
    control.execute(vehicle, _ctx())
    store = vehicle.store
    want = atan2(-GRAV, -0.0)
    phiavx_deg = want * DEG
    if abs(phiavx_deg) >= 90.0:
        phiavx_deg = 90.0 * _sign(phiavx_deg)
    assert _approx(store.get("phiav"), want)
    assert _approx(store.get("phiavcx"), want * DEG)
    assert _approx(store.get("phiavx"), phiavx_deg)
    assert _approx(store.get("phiavout"), phiavx_deg * RAD)
    assert _approx(store.get("anx"), 1.0)


def test_identity_tvl_lateral_command_bank_45_deg():
    vehicle, control = _ready(
        acoml=(0.0, GRAV, -GRAV),
        tvl=IDENTITY,
        tphi=0.0,
        tanx=0.0,
        philimx=90.0,
        acft_option=0,
        grav=GRAV,
    )
    control.execute(vehicle, _ctx())
    store = vehicle.store
    want = atan2(GRAV, GRAV)
    assert _approx(store.get("phiavcx"), want * DEG)
    assert _approx(store.get("phiavx"), want * DEG)
    assert _approx(store.get("phiavout"), want * DEG * RAD)
    assert _approx(store.get("ancomx"), sqrt(2.0))
    assert _approx(store.get("anx"), sqrt(2.0))


def test_tphi_zero_no_lag():
    vehicle, control = _ready(
        acoml=(0.0, GRAV, -GRAV),
        tvl=IDENTITY,
        tphi=0.0,
        tanx=0.0,
        philimx=90.0,
        acft_option=0,
        grav=GRAV,
        phiav=1.0,
        phiavd=-3.0,
    )
    control.execute(vehicle, _ctx())
    store = vehicle.store
    want = atan2(GRAV, GRAV)
    assert _approx(store.get("phiav"), want)
    assert _approx(store.get("phiavd"), -3.0)


def test_tphi_lag_stored_slope():
    int_step = 0.002
    vehicle, control = _ready(
        acoml=(0.0, GRAV, -GRAV),
        tvl=IDENTITY,
        tphi=0.5,
        tanx=0.0,
        philimx=90.0,
        acft_option=0,
        grav=GRAV,
        phiav=0.0,
        phiavd=0.0,
    )
    want = _expected(vehicle.store, int_step)
    control.execute(vehicle, _ctx(int_step=int_step))
    store = vehicle.store
    assert _approx(store.get("phiav"), want["phiav"])
    assert _approx(store.get("phiavd"), want["phiavd"])
    assert _approx(store.get("phiavx"), want["phiavx"])
    assert _approx(store.get("phiavout"), want["phiavout"])
    assert store.get("phiav") != pytest.approx(atan2(GRAV, GRAV), rel=RTOL, abs=ATOL)


def test_philimx_limits_phiavx_and_phiavout_not_phiav_state():
    vehicle, control = _ready(
        acoml=(0.0, GRAV, -GRAV),
        tvl=IDENTITY,
        tphi=0.0,
        tanx=0.0,
        philimx=30.0,
        acft_option=0,
        grav=GRAV,
    )
    control.execute(vehicle, _ctx())
    store = vehicle.store
    unlimited = atan2(GRAV, GRAV)
    assert _approx(store.get("phiav"), unlimited)
    assert _approx(store.get("phiavx"), 30.0)
    assert _approx(store.get("phiavout"), 30.0 * RAD)
    assert abs(store.get("phiavx")) <= 30.0


def test_negative_bank_limiter_uses_cadac_sign():
    vehicle, control = _ready(
        acoml=(0.0, -GRAV, -GRAV),
        tvl=IDENTITY,
        tphi=0.0,
        tanx=0.0,
        philimx=30.0,
        acft_option=0,
        grav=GRAV,
    )
    control.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("phiavx"), -30.0)
    assert _approx(store.get("phiavout"), -30.0 * RAD)


def test_tanx_zero_anx_equals_ancomx():
    vehicle, control = _ready(
        acoml=(0.0, GRAV, -GRAV),
        tvl=IDENTITY,
        tphi=0.0,
        tanx=0.0,
        philimx=90.0,
        acft_option=0,
        grav=GRAV,
        anx=0.25,
        anxd=4.0,
    )
    control.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("anx"), sqrt(2.0))
    assert _approx(store.get("ancomx"), sqrt(2.0))
    assert _approx(store.get("anxd"), 4.0)


def test_tanx_lag_stored_slope():
    int_step = 0.002
    vehicle, control = _ready(
        acoml=(0.0, GRAV, -GRAV),
        tvl=IDENTITY,
        tphi=0.0,
        tanx=0.4,
        philimx=90.0,
        acft_option=0,
        grav=GRAV,
        anx=0.0,
        anxd=0.0,
    )
    want = _expected(vehicle.store, int_step)
    control.execute(vehicle, _ctx(int_step=int_step))
    store = vehicle.store
    assert _approx(store.get("anx"), want["anx"])
    assert _approx(store.get("anxd"), want["anxd"])
    assert _approx(store.get("ancomx"), want["ancomx"])
    assert store.get("anx") != pytest.approx(sqrt(2.0), rel=RTOL, abs=ATOL)


def test_option_positive_alpha_limiter():
    pdynmc = 20000.0
    clalpha = 0.0523
    alplimx = 12.0
    wingloading = 3247.0
    anlimx = pdynmc * clalpha * alplimx / wingloading
    vehicle, control = _ready(
        acoml=(0.0, 100.0, -100.0),
        tvl=IDENTITY,
        tphi=0.0,
        tanx=0.0,
        philimx=90.0,
        acft_option=1,
        grav=GRAV,
        pdynmc=pdynmc,
        clalpha=clalpha,
        alplimx=alplimx,
        wingloading=wingloading,
    )
    control.execute(vehicle, _ctx())
    store = vehicle.store
    ancomx = sqrt(100.0 * 100.0 + 100.0 * 100.0) / GRAV
    assert ancomx > anlimx
    assert _approx(store.get("ancomx"), ancomx)
    assert _approx(store.get("anx"), anlimx)


def test_option_zero_skips_alpha_limiter():
    pdynmc = 20000.0
    clalpha = 0.0523
    alplimx = 12.0
    wingloading = 3247.0
    anlimx = pdynmc * clalpha * alplimx / wingloading
    vehicle, control = _ready(
        acoml=(0.0, 100.0, -100.0),
        tvl=IDENTITY,
        tphi=0.0,
        tanx=0.0,
        philimx=90.0,
        acft_option=0,
        grav=GRAV,
        pdynmc=pdynmc,
        clalpha=clalpha,
        alplimx=alplimx,
        wingloading=wingloading,
    )
    control.execute(vehicle, _ctx())
    store = vehicle.store
    ancomx = sqrt(100.0 * 100.0 + 100.0 * 100.0) / GRAV
    assert ancomx > anlimx
    assert _approx(store.get("anx"), ancomx)


def test_near_zero_acomv_bank_is_zero():
    tiny = EPS * 0.1
    vehicle, control = _ready(
        acoml=(0.0, tiny, tiny),
        tvl=IDENTITY,
        tphi=0.0,
        tanx=0.0,
        philimx=90.0,
        acft_option=0,
        grav=GRAV,
    )
    control.execute(vehicle, _ctx())
    store = vehicle.store
    assert _approx(store.get("phiav"), 0.0)
    assert _approx(store.get("phiavcx"), 0.0)
    assert _approx(store.get("phiavout"), 0.0)
    assert _approx(store.get("ancomx"), sqrt(tiny * tiny + tiny * tiny) / GRAV)


def test_cadac_sign_zero_is_plus_one_not_numpy_sign():
    from cadac.vehicles.flat6.sam6.aircraft import _sign as prod_sign

    assert prod_sign(0.0) == 1
    assert prod_sign(-0.0) == 1
    assert prod_sign(-1.0) == -1
    assert prod_sign(1.0) == 1
    assert np.sign(0.0) == 0.0


def test_terminate_is_pass():
    vehicle, control = _ready()
    assert control.terminate(vehicle, _ctx()) is None
    assert _approx(vehicle.store.get("phiavout"), 0.0)
    assert _approx(vehicle.store.get("anx"), 0.0)


def test_no_flat6_or_plane_imports():
    import cadac.vehicles.flat6.sam6.aircraft as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "cadac.eom.flat6" not in src
    assert "Flat6" not in src
    assert "plane5" not in src
    assert "plane6" not in src
    assert "Plane5" not in src
    assert "Plane6" not in src
    assert "hyper5" not in src
    assert "hyper6" not in src
    assert "np.sign" not in src
    assert "_cadac_sign" not in src

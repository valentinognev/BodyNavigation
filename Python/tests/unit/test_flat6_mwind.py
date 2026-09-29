"""Flat6Environment mwind=1 (constant) and mwind=2 (shear) — FALCON6 laws."""

from math import cos, sin
from types import SimpleNamespace

import numpy as np
import pytest

from cadac.constants import RAD
from cadac.env.us76 import atmosphere76
from cadac.eom.flat6 import Flat6Environment
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore

RTOL = 1e-12
ATOL = 1e-14

DT = 0.001
TWIND = 0.1
DVAE = 5.0
PSIWDX = 0.0
VAED3 = 0.0
HBE = 1000.0
VBEL = np.array([180.0, 0.0, 0.0], dtype=float)

# Shear profile: mid-band → linear interpolate
DVAEL = 5.0
WALTL = 0.0
DVAEH = 15.0
WALTH = 2000.0


def _ctx(int_step=DT):
    return SimpleNamespace(int_step=int_step)


def _vehicle(
    *,
    mwind,
    hbe=HBE,
    vbel=VBEL,
    dvae=DVAE,
    psiwdx=PSIWDX,
    vaed3=VAED3,
    twind=TWIND,
    dvael=DVAEL,
    waltl=WALTL,
    dvaeh=DVAEH,
    walth=WALTH,
):
    s = StateStore()
    vehicle = SimpleNamespace(store=s)
    env = Flat6Environment()
    env.define(vehicle)
    s.define(Field("hbe", hbe, "real", "out", "newton"))
    s.define(Field("VBEL", vbel, "vec", "diag", "newton"))
    s.set("mwind", mwind)
    s.set("dvae", dvae)
    s.set("psiwdx", psiwdx)
    s.set("vaed3", vaed3)
    s.set("twind", twind)
    s.set("dvael", dvael)
    s.set("waltl", waltl)
    s.set("dvaeh", dvaeh)
    s.set("walth", walth)
    return vehicle, env


def _smoothed_vael(dvw, psiwdx, vaed3, twind, dt):
    vael_raw = np.array(
        [
            -dvw * cos(psiwdx * RAD),
            -dvw * sin(psiwdx * RAD),
            vaed3,
        ],
        dtype=float,
    )
    vaels = np.zeros(3)
    vaelsd = np.zeros(3)
    vaelsd_new = (vael_raw - vaels) * (1.0 / twind)
    return integrate(vaelsd_new, vaelsd, vaels, dt)


def test_flat6_mwind_1_constant():
    """Constant wind: mwind=1, dvae magnitude, smoothed VAEL."""
    vehicle, env = _vehicle(mwind=1)
    env.execute(vehicle, _ctx())
    store = vehicle.store
    want = _smoothed_vael(DVAE, PSIWDX, VAED3, TWIND, DT)
    np.testing.assert_allclose(store.get("VAEL"), want, rtol=RTOL, atol=ATOL)
    assert store.get("VAEL")[0] < 0.0
    np.testing.assert_allclose(
        store.get("VBAL"), VBEL - store.get("VAEL"), rtol=RTOL, atol=ATOL
    )
    dvba = float(np.linalg.norm(store.get("VBAL")))
    np.testing.assert_allclose(store.get("dvba"), dvba, rtol=RTOL, atol=ATOL)
    rho, _, _ = atmosphere76(HBE)
    np.testing.assert_allclose(store.get("rho"), rho, rtol=RTOL, atol=ATOL)


def test_flat6_mwind_2_shear():
    """Shear wind: mwind=2, linear speed between waltl/walth, then smoothed."""
    vehicle, env = _vehicle(mwind=2)
    env.execute(vehicle, _ctx())
    store = vehicle.store
    dvw = DVAEL + (DVAEH - DVAEL) * (HBE - WALTL) / (WALTH - WALTL)
    want = _smoothed_vael(dvw, PSIWDX, VAED3, TWIND, DT)
    np.testing.assert_allclose(store.get("VAEL"), want, rtol=RTOL, atol=ATOL)
    assert abs(dvw - 10.0) < 1e-12
    assert store.get("VAEL")[0] < 0.0
    np.testing.assert_allclose(
        store.get("VBAL"), VBEL - store.get("VAEL"), rtol=RTOL, atol=ATOL
    )


def test_flat6_unknown_mwind_raises():
    """mwind 1/2 are supported; other nonzero values still raise."""
    vehicle, env = _vehicle(mwind=3)
    with pytest.raises(ValueError, match="mwind"):
        env.execute(vehicle, _ctx())

"""FALCON6 Flat6Environment mguid==6 termination and mfreeze env latch."""

from types import SimpleNamespace

import numpy as np
import pytest

from cadac.eom.flat6 import Flat6Environment
from cadac.kernel.state import Field, StateStore

HBE = 1000.0
RTOL = 1e-12
ATOL = 1e-12


def _vehicle(*, vbel, mguid=6, trmach=0.5, trdynm=0.0, trcode=0.0, mwind=0, mfreeze=None):
    s = StateStore()
    vehicle = SimpleNamespace(store=s)
    env = Flat6Environment()
    env.define(vehicle)
    s.define(Field("hbe", HBE, "real", "out", "newton"))
    s.define(Field("VBEL", np.array(vbel, dtype=float), "vec", "state", "newton"))
    s.define(Field("mguid", mguid, "int", "data", "guidance"))
    s.define(Field("trmach", trmach, "real", "data", "aerodynamics"))
    s.define(Field("trdynm", trdynm, "real", "data", "aerodynamics"))
    s.define(Field("trcode", trcode, "real", "init", "aerodynamics"))
    s.set("mwind", mwind)
    if mfreeze is not None:
        s.define(Field("mfreeze", mfreeze, "int", "data", "control"))
    return vehicle, env


def test_falcon6_env_mguid6_sets_trcode_2_on_low_mach():
    """C++: mguid==6 and vmach<=trmach → trcode=2 (pdynmc above trdynm)."""
    vehicle, env = _vehicle(vbel=(1.0, 0.0, 0.0), mguid=6, trmach=0.5, trdynm=0.0)
    env.execute(vehicle, None)
    assert vehicle.store.get("trcode") == 2.0
    assert vehicle.store.get("vmach") <= 0.5


def test_falcon6_mfreeze_holds_vmach_pdynmc():
    """C++: mfreeze=1 latches vmach/pdynmc; mfreeze=0 clears latch."""
    vehicle, env = _vehicle(vbel=(180.0, 0.0, 0.0), mguid=0, mfreeze=1)
    store = vehicle.store
    env.execute(vehicle, None)
    frozen_mach = store.get("vmach")
    frozen_q = store.get("pdynmc")
    assert store.get("mfreeze_environ") == 1
    assert store.get("vmachf") == pytest.approx(frozen_mach, rel=RTOL, abs=ATOL)
    assert store.get("pdynmcf") == pytest.approx(frozen_q, rel=RTOL, abs=ATOL)

    store.set("VBEL", np.array([32.0, 0.0, 0.0], dtype=float))
    env.execute(vehicle, None)
    assert store.get("vmach") == pytest.approx(frozen_mach, rel=RTOL, abs=ATOL)
    assert store.get("pdynmc") == pytest.approx(frozen_q, rel=RTOL, abs=ATOL)
    assert store.get("mfreeze_environ") == 1

    store.set("mfreeze", 0)
    env.execute(vehicle, None)
    assert store.get("mfreeze_environ") == 0
    assert store.get("vmach") != pytest.approx(frozen_mach, rel=RTOL, abs=ATOL)
    assert store.get("pdynmc") != pytest.approx(frozen_q, rel=RTOL, abs=ATOL)

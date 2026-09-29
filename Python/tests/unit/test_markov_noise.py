"""Per-step markov_noise before modules (C++ execution.cpp)."""

from types import SimpleNamespace

from cadac.kernel.executive import run_loop
from cadac.kernel.state import Field, StateStore
from cadac.stoch import seed


def _entry(name, sigma, bcor):
    return SimpleNamespace(
        name=name, sigma=sigma, bcor=bcor, saved=0.0, status=True
    )


class _Vehicle:
    def __init__(self, markov_list=None):
        self.store = StateStore()
        self.store.define(Field("time", 0.0, "real", "exec", "environment"))
        self.store.define(Field("randt", 0.0, "real", "data", "sensor"))
        self.events = None
        self.event_time = 0.0
        self.markov_list = markov_list or []


class _Watch:
    name = "watch"

    def __init__(self, seen):
        self.seen = seen

    def define(self, vehicle):
        pass

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        self.seen.append(vehicle.store.get("randt"))


def test_markov_noise_called_each_step():
    """C++: markov_noise before modules each step; nmonte>0 keeps live samples."""
    seed(12345)
    entries = [_entry("randt", 0.0005, 100.0)]
    vehicle = _Vehicle(markov_list=entries)
    seen = []
    run_loop(
        vehicles=[vehicle],
        modules_by_vehicle={vehicle: [_Watch(seen)]},
        module_order=["watch"],
        end_time=0.2,
        int_step=0.1,
        nmonte=1,
    )
    # four integration ticks: 0.0, 0.1, 0.2, 0.3
    assert len(seen) == 4
    assert all(v != 0.0 for v in seen)
    # correlated Markov path is not a constant hold of the first draw
    assert len({round(v, 12) for v in seen}) >= 2
    assert entries[0].saved != 0.0


def test_markov_noise_nmonte0_draw_then_zero():
    """nmonte==0: still draw (advance saved), then force store to 0 before modules."""
    seed(12345)
    entries = [_entry("randt", 0.0005, 100.0)]
    vehicle = _Vehicle(markov_list=entries)
    seen = []
    run_loop(
        vehicles=[vehicle],
        modules_by_vehicle={vehicle: [_Watch(seen)]},
        module_order=["watch"],
        end_time=0.1,
        int_step=0.1,
        nmonte=0,
    )
    assert seen == [0.0, 0.0, 0.0]
    assert vehicle.store.get("randt") == 0.0
    assert entries[0].saved != 0.0

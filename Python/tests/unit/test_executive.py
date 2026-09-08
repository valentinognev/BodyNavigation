import numpy as np

from cadac.kernel.events import EventEngine, EventSpec
from cadac.kernel.executive import SimContext, run_loop
from cadac.kernel.module import DummyModule
from cadac.kernel.state import Field, StateStore


class _Vehicle:
    def __init__(self, events=None):
        self.store = StateStore()
        self.store.define(Field("time", 0.0, "real", "exec", "environment"))
        self.events = EventEngine(events or [])
        self.event_time = 0.0


def test_loop_times_start_of_iteration():
    vehicle = _Vehicle()
    dummy = DummyModule()
    times = run_loop(
        vehicles=[vehicle],
        modules_by_vehicle={vehicle: [dummy]},
        module_order=["dummy"],
        end_time=0.2,
        int_step=0.1,
    )
    np.testing.assert_allclose(
        times, [0.0, 0.1, 0.2, 0.3], rtol=1e-12, atol=1e-14
    )


def test_dummy_module_sets_store_time():
    vehicle = _Vehicle()
    run_loop(
        vehicles=[vehicle],
        modules_by_vehicle={vehicle: [DummyModule()]},
        module_order=["dummy"],
        end_time=0.2,
        int_step=0.1,
    )
    np.testing.assert_allclose(vehicle.store.get("time"), 0.3, rtol=1e-12, atol=1e-14)


def test_dummy_execute_writes_ctx_sim_time():
    vehicle = _Vehicle()
    ctx = SimContext(
        sim_time=1.5,
        int_step=0.1,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )
    DummyModule().execute(vehicle, ctx)
    assert vehicle.store.get("time") == 1.5


def test_health_not_one_skips_modules():
    vehicle = _Vehicle()
    vehicle.health = 0
    run_loop(
        vehicles=[vehicle],
        modules_by_vehicle={vehicle: [DummyModule()]},
        module_order=["dummy"],
        end_time=0.2,
        int_step=0.1,
    )
    assert vehicle.store.get("time") == 0.0
    np.testing.assert_allclose(vehicle.event_time, 0.4, rtol=1e-12, atol=1e-14)


def test_status_not_one_skips_modules():
    vehicle = _Vehicle()
    vehicle.status = -1
    run_loop(
        vehicles=[vehicle],
        modules_by_vehicle={vehicle: [DummyModule()]},
        module_order=["dummy"],
        end_time=0.2,
        int_step=0.1,
    )
    assert vehicle.store.get("time") == 0.0


def test_events_evaluate_before_modules():
    vehicle = _Vehicle(
        events=[EventSpec(when={"time": {">": -1}}, set={"mprop": 2})]
    )
    vehicle.store.define(Field("mprop", 1, "int", "data", "propulsion"))
    seen = []

    class _Watch:
        name = "watch"

        def define(self, vehicle):
            pass

        def initialize(self, vehicle, ctx):
            pass

        def execute(self, vehicle, ctx):
            seen.append(vehicle.store.get("mprop"))

        def terminate(self, vehicle, ctx):
            pass

    run_loop(
        vehicles=[vehicle],
        modules_by_vehicle={vehicle: [_Watch()]},
        module_order=["watch"],
        end_time=0.0,
        int_step=0.1,
    )
    assert seen[0] == 2


def test_event_epoch_resets_event_time():
    vehicle = _Vehicle(
        events=[EventSpec(when={"time": {">": -1}}, set={"mprop": 2})]
    )
    vehicle.store.define(Field("mprop", 1, "int", "data", "propulsion"))
    vehicle.event_time = 5.0
    seen = []

    class _Watch:
        name = "watch"

        def define(self, vehicle):
            pass

        def initialize(self, vehicle, ctx):
            pass

        def execute(self, vehicle, ctx):
            seen.append(ctx.event_time)

        def terminate(self, vehicle, ctx):
            pass

    run_loop(
        vehicles=[vehicle],
        modules_by_vehicle={vehicle: [_Watch()]},
        module_order=["watch"],
        end_time=0.0,
        int_step=0.1,
    )
    assert seen[0] == 0.0


def test_modules_follow_module_order():
    vehicle = _Vehicle()
    log = []

    class _Named:
        def __init__(self, name):
            self.name = name

        def define(self, vehicle):
            pass

        def initialize(self, vehicle, ctx):
            pass

        def execute(self, vehicle, ctx):
            log.append(self.name)

        def terminate(self, vehicle, ctx):
            pass

    first, second = _Named("first"), _Named("second")
    run_loop(
        vehicles=[vehicle],
        modules_by_vehicle={vehicle: [second, first]},
        module_order=["first", "second"],
        end_time=0.0,
        int_step=0.1,
    )
    assert log == ["first", "second", "first", "second"]


def test_run_loop_adopts_ctx_int_step():
    vehicle = _Vehicle()

    class _Resize:
        name = "resize"

        def define(self, vehicle):
            pass

        def initialize(self, vehicle, ctx):
            pass

        def execute(self, vehicle, ctx):
            ctx.int_step = 0.05

        def terminate(self, vehicle, ctx):
            pass

    times = run_loop(
        vehicles=[vehicle],
        modules_by_vehicle={vehicle: [_Resize()]},
        module_order=["resize"],
        end_time=0.1,
        int_step=0.1,
    )
    np.testing.assert_allclose(
        times, [0.0, 0.05, 0.10, 0.15], rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(vehicle.event_time, 0.20, rtol=1e-12, atol=1e-14)


def test_skip_module_absent_on_vehicle():
    a = _Vehicle()
    b = _Vehicle()
    dummy = DummyModule()
    run_loop(
        vehicles=[a, b],
        modules_by_vehicle={a: [dummy], b: []},
        module_order=["dummy"],
        end_time=0.0,
        int_step=0.1,
    )
    assert a.store.get("time") == 0.1


def test_run_loop_does_not_rebuild_name_map_each_step():
    vehicle = _Vehicle()
    calls = {"maps": 0}

    class _Named:
        name = "watch"

        def define(self, vehicle):
            pass

        def initialize(self, vehicle, ctx):
            pass

        def execute(self, vehicle, ctx):
            pass

        def terminate(self, vehicle, ctx):
            pass

    class _Dict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __getitem__(self, key):
            if key is vehicle or key == vehicle:
                calls["maps"] += 1
            return super().__getitem__(key)

    dummy = _Named()
    modules = _Dict({vehicle: [dummy]})
    run_loop(
        vehicles=[vehicle],
        modules_by_vehicle=modules,
        module_order=["watch"],
        end_time=0.2,
        int_step=0.1,
    )
    # 4 time stations (0, 0.1, 0.2, 0.3). Lookup once at bind, not once per step.
    assert calls["maps"] == 1


def test_run_loop_reuses_simcontext_per_vehicle():
    vehicle = _Vehicle()
    ids = []

    class _Watch:
        name = "watch"

        def define(self, vehicle):
            pass

        def initialize(self, vehicle, ctx):
            pass

        def execute(self, vehicle, ctx):
            ids.append(id(ctx))

        def terminate(self, vehicle, ctx):
            pass

    run_loop(
        vehicles=[vehicle],
        modules_by_vehicle={vehicle: [_Watch()]},
        module_order=["watch"],
        end_time=0.1,
        int_step=0.1,
    )
    assert len(ids) >= 2
    assert len(set(ids)) == 1

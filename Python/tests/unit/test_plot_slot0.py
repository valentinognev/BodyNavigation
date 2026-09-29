from cadac.cli import make_plot_on_step, track_columns, track_names
from cadac.kernel.executive import run_loop
from cadac.kernel.module import DummyModule
from cadac.kernel.state import Field, StateStore
from test_executive import _Vehicle


def test_tracks_use_newton_sael_when_sbel_is_not_newton():
    target = _Vehicle()
    target.name = "Target"
    target.store.define(Field("SBEL", (0.0, 0.0, 0.0), "vec", "state", "target"))
    target.store.define(
        Field("SAEL", (10000.0, 500.0, -2000.0), "vec", "state", "newton", ("com",))
    )
    plot_rows = []
    tracks = [[] for _ in range(1)]
    int_step = 0.1
    on_step = make_plot_on_step(
        plot_rows,
        plot_step=int_step,
        nveh=1,
        columns_fn=lambda vehicle: ["time"],
        tracks=tracks,
    )
    run_loop(
        vehicles=[target],
        modules_by_vehicle={target: [DummyModule()]},
        module_order=["dummy"],
        end_time=0.0,
        int_step=int_step,
        on_step=on_step,
    )
    assert tracks[0][0]["SAEL1"] == 10000.0
    assert tracks[0][0]["SAEL2"] == 500.0
    assert tracks[0][0]["SAEL3"] == -2000.0
    assert "SBEL1" not in tracks[0][0]


def test_tracks_keep_sbel_when_both_positions_are_newton():
    vehicle = _Vehicle()
    vehicle.name = "Missile"
    vehicle.store.define(Field("SBEL", (1.0, 2.0, 3.0), "vec", "state", "newton"))
    vehicle.store.define(Field("SAEL", (9.0, 9.0, 9.0), "vec", "state", "newton"))
    plot_rows = []
    tracks = [[] for _ in range(1)]
    int_step = 0.1
    on_step = make_plot_on_step(
        plot_rows,
        plot_step=int_step,
        nveh=1,
        columns_fn=lambda vehicle: ["time"],
        tracks=tracks,
    )
    run_loop(
        vehicles=[vehicle],
        modules_by_vehicle={vehicle: [DummyModule()]},
        module_order=["dummy"],
        end_time=0.0,
        int_step=int_step,
        on_step=on_step,
    )
    assert tracks[0][0]["SBEL1"] == 1.0
    assert "SAEL1" not in tracks[0][0]


def test_track_columns_keeps_non_newton_sbel_when_alone():
    store = StateStore()
    store.define(Field("SBEL", (1.0, 2.0, 3.0), "vec", "state", "trajectory"))
    assert track_columns(store) == ["SBEL1", "SBEL2", "SBEL3"]


def test_plot_rows_only_vehicle_slot_0():
    a = _Vehicle()
    b = _Vehicle()
    dummy = DummyModule()
    plot_rows = []
    int_step = 0.1
    on_step = make_plot_on_step(
        plot_rows, plot_step=int_step, nveh=2, columns_fn=lambda vehicle: ["time"]
    )
    run_loop(
        vehicles=[a, b],
        modules_by_vehicle={a: [dummy], b: [dummy]},
        module_order=["dummy"],
        end_time=0.2,
        int_step=int_step,
        on_step=on_step,
    )
    ticks = 4
    assert len(plot_rows) == ticks
    assert len(plot_rows) != 2 * ticks


def test_tracks_record_every_vehicle_position():
    missile = _Vehicle()
    missile.name = "Missile"
    missile.store.define(Field("SBEL", (0.0, 1.0, -2.0), "vec", "state", "newton", ("plot",)))
    target = _Vehicle()
    target.name = "Target"
    target.store.define(Field("SAEL", (3.0, 4.0, -5.0), "vec", "state", "newton", ("com",)))
    plot_rows = []
    tracks = [[] for _ in range(2)]
    int_step = 0.1
    on_step = make_plot_on_step(
        plot_rows,
        plot_step=int_step,
        nveh=2,
        columns_fn=lambda vehicle: ["time"],
        tracks=tracks,
    )
    run_loop(
        vehicles=[missile, target],
        modules_by_vehicle={missile: [DummyModule()], target: [DummyModule()]},
        module_order=["dummy"],
        end_time=0.2,
        int_step=int_step,
        on_step=on_step,
    )
    assert len(plot_rows) == 4
    assert tracks[0][0]["SBEL1"] == 0.0
    assert tracks[0][0]["SBEL2"] == 1.0
    assert tracks[0][0]["SBEL3"] == -2.0
    assert tracks[1][0]["SAEL1"] == 3.0
    assert tracks[1][0]["SAEL2"] == 4.0
    assert tracks[1][0]["SAEL3"] == -5.0
    assert len(tracks[0]) == 4
    assert len(tracks[1]) == 4


def test_tracks_keep_each_vehicles_plot_columns():
    missile = _Vehicle()
    missile.name = "Missile"
    missile.store.define(Field("dvbe", 10.0, "real", "out", "newton", ("plot",)))
    missile.store.define(Field("SBEL", (0.0, 1.0, -2.0), "vec", "state", "newton", ("plot",)))
    target = _Vehicle()
    target.name = "Target"
    target.store.define(Field("dvbe", 20.0, "real", "out", "newton", ("plot",)))
    target.store.define(Field("mach", 0.8, "real", "out", "environment", ("plot",)))
    target.store.define(Field("SAEL", (3.0, 4.0, -5.0), "vec", "state", "newton", ("com",)))

    def columns(vehicle):
        names = ["time"]
        for name in ("dvbe", "mach"):
            if name in vehicle.store:
                names.append(name)
        return names

    plot_rows = []
    tracks = [[] for _ in range(2)]
    int_step = 0.1
    on_step = make_plot_on_step(
        plot_rows,
        plot_step=int_step,
        nveh=2,
        columns_fn=columns,
        tracks=tracks,
    )
    run_loop(
        vehicles=[missile, target],
        modules_by_vehicle={missile: [DummyModule()], target: [DummyModule()]},
        module_order=["dummy"],
        end_time=0.0,
        int_step=int_step,
        on_step=on_step,
    )
    assert tracks[0][0]["time"] == 0.0
    assert tracks[0][0]["dvbe"] == 10.0
    assert "mach" not in tracks[0][0]
    assert tracks[0][0]["SBEL2"] == 1.0
    assert tracks[1][0]["dvbe"] == 20.0
    assert tracks[1][0]["mach"] == 0.8
    assert tracks[1][0]["SAEL1"] == 3.0
    assert len(plot_rows) == 2
    assert "SAEL1" not in plot_rows[0]


def test_duplicate_vehicle_names_are_numbered():
    assert track_names(["Missile", "Missile", "Target", "Target"]) == [
        "Missile 1",
        "Missile 2",
        "Target 1",
        "Target 2",
    ]

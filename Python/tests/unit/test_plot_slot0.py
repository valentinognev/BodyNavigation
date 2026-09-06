from cadac.cli import make_plot_on_step
from cadac.kernel.executive import run_loop
from cadac.kernel.module import DummyModule
from test_executive import _Vehicle


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

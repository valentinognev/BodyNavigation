from cadac.env.gravity import gravity
from cadac.env.iso62 import iso62
from cadac.eom.round3 import Round3Environment
from cadac.kernel.executive import SimContext
from cadac.kernel.state import Field, StateStore


class _Vehicle:
    def __init__(self):
        self.store = StateStore()


def _ctx(**kwargs):
    fields = dict(
        sim_time=0.0,
        int_step=0.1,
        event_time=0.0,
        out_fact=0.0,
        combus=None,
        vehicle_slot=0,
    )
    fields.update(kwargs)
    return SimContext(**fields)


def _defined_env():
    vehicle = _Vehicle()
    env = Round3Environment()
    env.define(vehicle)
    return vehicle, env


def test_name_is_environment():
    assert Round3Environment().name == "environment"


def test_define_registers_spec_fields():
    vehicle, env = _defined_env()
    for name in (
        "time",
        "event_time",
        "int_step_new",
        "out_step_fact",
        "grav",
        "rho",
        "pdynmc",
        "mach",
        "vsound",
        "press",
    ):
        assert vehicle.store.get(name) == 0.0


def test_initialize_sets_time_and_int_step_new():
    vehicle, env = _defined_env()
    env.initialize(vehicle, _ctx(sim_time=3.0, int_step=0.2))
    assert vehicle.store.get("time") == 3.0
    assert vehicle.store.get("int_step_new") == 0.2


def test_execute_mach_pdynmc_grav_match_iso62_and_gravity():
    vehicle, env = _defined_env()
    vehicle.store.define(Field("alt", 3000.0, "real", "state", "newton"))
    vehicle.store.define(Field("dvbe", 250.0, "real", "state", "newton"))
    env.execute(vehicle, _ctx())
    atm = iso62(3000.0, 250.0)
    assert vehicle.store.get("mach") == atm["mach"]
    assert vehicle.store.get("pdynmc") == atm["pdynmc"]
    assert vehicle.store.get("rho") == atm["rho"]
    assert vehicle.store.get("vsound") == atm["vsound"]
    assert vehicle.store.get("press") == atm["press"]
    assert vehicle.store.get("grav") == gravity(3000.0)


def test_execute_copies_sim_time_and_event_time():
    vehicle, env = _defined_env()
    vehicle.store.define(Field("alt", 3000.0, "real", "state", "newton"))
    vehicle.store.define(Field("dvbe", 250.0, "real", "state", "newton"))
    env.execute(vehicle, _ctx(sim_time=1.5, event_time=0.4))
    assert vehicle.store.get("time") == 1.5
    assert vehicle.store.get("event_time") == 0.4


def test_execute_sets_ctx_int_step_and_out_fact():
    vehicle, env = _defined_env()
    vehicle.store.define(Field("alt", 3000.0, "real", "state", "newton"))
    vehicle.store.define(Field("dvbe", 250.0, "real", "state", "newton"))
    vehicle.store.set("int_step_new", 0.05)
    vehicle.store.set("out_step_fact", 2.0)
    ctx = _ctx(int_step=0.1, out_fact=0.0)
    env.execute(vehicle, ctx)
    assert ctx.int_step == 0.05
    assert ctx.out_fact == 2.0

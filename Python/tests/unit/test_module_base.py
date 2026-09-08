from cadac.kernel.module import ModuleBase
from cadac.kernel.state import Field, StateStore


class _Env(ModuleBase):
    name = "environment"
    fields = (
        Field("press", 0.0, "real", "out", "environment"),
    )

    def execute(self, vehicle, ctx):
        vehicle.store.set("press", 1.0)


def test_module_base_define_does_not_share_field_objects():
    a, b = type("V", (), {})(), type("V", (), {})()
    a.store, b.store = StateStore(), StateStore()
    mod = _Env()
    mod.define(a)
    mod.define(b)
    a.store.set("press", 3.0)
    assert b.store.get("press") == 0.0


def test_module_base_terminate_is_noop():
    v = type("V", (), {})()
    v.store = StateStore()
    _Env().define(v)
    _Env().terminate(v, None)
    assert v.store.get("press") == 0.0


def test_module_base_execute_raises_if_not_overridden():
    class _Bare(ModuleBase):
        name = "bare"
        fields = ()

    v = type("V", (), {})()
    v.store = StateStore()
    try:
        _Bare().execute(v, None)
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError")

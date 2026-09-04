from cadac.kernel.state import Field, StateStore
from cadac.kernel.events import EventEngine, EventSpec


def test_time_then_set():
    s = StateStore()
    s.define(Field("time", 0.0, "real", "exec", "environment"))
    s.define(Field("mprop", 1, "int", "data", "propulsion"))
    eng = EventEngine([EventSpec(when={"time": {">": 10}}, set={"mprop": 2})])
    s.set("time", 10.0)
    assert eng.evaluate(s) is False
    s.set("time", 10.01)
    assert eng.evaluate(s) is True
    assert s.get("mprop") == 2
    s.set("time", 11.0)
    assert eng.evaluate(s) is False


def test_var_op_equals():
    s = StateStore()
    s.define(Field("wp_flag", 0, "int", "data", "guidance"))
    s.define(Field("mprop", 1, "int", "data", "propulsion"))
    eng = EventEngine(
        [EventSpec(when={"var": "wp_flag", "op": "=", "value": -1}, set={"mprop": 2})]
    )
    assert eng.evaluate(s) is False
    s.set("wp_flag", -1)
    assert eng.evaluate(s) is True
    assert s.get("mprop") == 2
    assert type(s.get("mprop")) is int
    assert eng.evaluate(s) is False


def test_less_than():
    s = StateStore()
    s.define(Field("time", 5.0, "real", "exec", "environment"))
    s.define(Field("mprop", 1, "int", "data", "propulsion"))
    eng = EventEngine([EventSpec(when={"time": {"<": 10}}, set={"mprop": 2})])
    assert eng.evaluate(s) is True
    assert s.get("mprop") == 2
    assert eng.evaluate(s) is False


def test_one_event_armed():
    s = StateStore()
    s.define(Field("time", 0.0, "real", "exec", "environment"))
    s.define(Field("mprop", 1, "int", "data", "propulsion"))
    s.define(Field("flag", 0, "int", "data", "guidance"))
    eng = EventEngine(
        [
            EventSpec(when={"time": {">": 10}}, set={"mprop": 2}),
            EventSpec(when={"time": {">": 5}}, set={"flag": 1}),
        ]
    )
    s.set("time", 6.0)
    assert eng.evaluate(s) is False
    assert s.get("flag") == 0
    s.set("time", 10.01)
    assert eng.evaluate(s) is True
    assert s.get("mprop") == 2
    assert s.get("flag") == 0
    assert eng.evaluate(s) is True
    assert s.get("flag") == 1
    assert eng.evaluate(s) is False

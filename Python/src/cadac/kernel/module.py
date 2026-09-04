from typing import Protocol


class Module(Protocol):
    name: str

    def define(self, vehicle) -> None: ...

    def initialize(self, vehicle, ctx) -> None: ...

    def execute(self, vehicle, ctx) -> None: ...

    def terminate(self, vehicle, ctx) -> None: ...


class DummyModule:
    name = "dummy"

    def define(self, vehicle):
        pass

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        vehicle.store.set("time", ctx.sim_time)

    def terminate(self, vehicle, ctx):
        pass

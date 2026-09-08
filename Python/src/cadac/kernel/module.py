from typing import Any, Protocol

from cadac.kernel.state import Field


class Module(Protocol):
    name: str

    def define(self, vehicle: Any) -> None: ...

    def initialize(self, vehicle: Any, ctx: Any) -> None: ...

    def execute(self, vehicle: Any, ctx: Any) -> None: ...

    def terminate(self, vehicle: Any, ctx: Any) -> None: ...


class ModuleBase:
    name: str
    fields: tuple = ()

    def define(self, vehicle: Any) -> None:
        store = vehicle.store
        for item in self.fields:
            if isinstance(item, Field):
                name, value, ftype, role, module, outputs = (
                    item.name,
                    item.value,
                    item.type,
                    item.role,
                    item.module,
                    item.outputs,
                )
            else:
                name, value, ftype, role, module, *rest = item
                outputs = rest[0] if rest else ()
            if name in store:
                continue
            store.define(Field(name, value, ftype, role, module, outputs))

    def initialize(self, vehicle: Any, ctx: Any) -> None:
        pass

    def execute(self, vehicle: Any, ctx: Any) -> None:
        raise NotImplementedError

    def terminate(self, vehicle: Any, ctx: Any) -> None:
        pass


class DummyModule(ModuleBase):
    name = "dummy"

    def execute(self, vehicle: Any, ctx: Any) -> None:
        vehicle.store.set("time", ctx.sim_time)

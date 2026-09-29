from cadac.kernel.state import Field


def stub_define(vehicle, module_name: str, fields: tuple[Field, ...] = ()) -> None:
    store = vehicle.store
    if not fields:
        fields = (Field(f"_{module_name}_stub", 0, "int", "save", module_name),)
    for field in fields:
        if field.name not in store:
            store.define(field)


class StubModule:
    """Skeleton module: define fields only; execute is a no-op until later tasks."""

    name: str
    _fields: tuple[Field, ...] = ()

    def define(self, vehicle):
        stub_define(vehicle, self.name, self._fields)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        pass

    def terminate(self, vehicle, ctx):
        pass

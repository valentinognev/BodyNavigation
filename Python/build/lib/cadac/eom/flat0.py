"""Zipfel 0-DOF ground-fixed kinematics (CADAC Flat0)."""

from cadac.kernel.module import ModuleBase
from cadac.kernel.state import Field


class Flat0Kinematics(ModuleBase):
    """Zipfel ground-fixed launch-time kinematics (CADAC ``flat0_kinematics``)."""

    name = "kinematics"
    fields = (
        Field("time", 0.0, "real", "out", "kinematics", ("com",)),
        Field("launch_delay", 0.0, "real", "data", "kinematics"),
        Field("launch_epoch", 0.0, "real", "init", "kinematics"),
        Field("launch_time", 0.0, "real", "out", "kinematics"),
    )

    def initialize(self, vehicle, ctx) -> None:
        store = vehicle.store
        store.set("time", ctx.sim_time)
        store.set("launch_epoch", store.get("launch_delay"))

    def execute(self, vehicle, ctx) -> None:
        store = vehicle.store
        store.set("launch_time", ctx.sim_time - store.get("launch_epoch"))
        store.set("time", ctx.sim_time)


class Flat0Newton(ModuleBase):
    """Zipfel ground-fixed relative position Newton (CADAC ``flat0_newton``)."""

    name = "newton"
    fields = (
        Field("srel1", 0.0, "real", "data", "newton"),
        Field("srel2", 0.0, "real", "data", "newton"),
        Field("srel3", 0.0, "real", "data", "newton"),
        Field("SREL", (0.0, 0.0, 0.0), "vec", "out", "newton"),
    )

    def initialize(self, vehicle, ctx) -> None:
        store = vehicle.store
        store.set(
            "SREL",
            (store.get("srel1"), store.get("srel2"), store.get("srel3")),
        )

    def execute(self, vehicle, ctx) -> None:
        pass

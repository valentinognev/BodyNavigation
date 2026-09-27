from cadac.kernel.state import Field


class Rocket6Intercept:
    name = "intercept"

    def define(self, vehicle):
        store = vehicle.store
        store.define(Field("write", 1, "int", "init", "intercept"))
        store.define(
            Field("modes", 0, "int", "diag", "intercept", ("scrn", "plot"))
        )

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        write = store.get("write")
        alt = store.get("alt")
        mguide = store.get("mguide")
        maut = store.get("maut")
        mprop = store.get("mprop")

        mauty = maut // 10
        mautp = maut % 10
        modes = mguide * 1000 + mauty * 100 + mautp * 10 + mprop

        if alt <= 0 and write:
            write = 0
            vehicle.health = 0
            if ctx.combus is not None:
                ctx.combus[ctx.vehicle_slot].status = 0

        store.set("write", write)
        store.set("modes", modes)

    def terminate(self, vehicle, ctx):
        pass

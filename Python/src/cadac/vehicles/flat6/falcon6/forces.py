import numpy as np

from cadac.kernel.state import Field


class Plane6Forces:
    name = "forces"

    def define(self, vehicle):
        zeros3 = (0.0, 0.0, 0.0)
        store = vehicle.store
        store.define(Field("FAPB", zeros3, "vec", "out", "forces"))
        store.define(Field("FMB", zeros3, "vec", "out", "forces"))

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        pdynmc = store.get("pdynmc")
        thrust = store.get("thrust")
        refa = store.get("refa")
        refb = store.get("refb")
        refc = store.get("refc")
        cxt = store.get("cxt")
        cyt = store.get("cyt")
        czt = store.get("czt")
        clt = store.get("clt")
        cmt = store.get("cmt")
        cnt = store.get("cnt")
        fapb = np.array(
            [
                pdynmc * refa * cxt + thrust,
                pdynmc * refa * cyt,
                pdynmc * refa * czt,
            ],
            dtype=float,
        )
        fmb = np.array(
            [
                pdynmc * refa * refb * clt,
                pdynmc * refa * refc * cmt,
                pdynmc * refa * refb * cnt,
            ],
            dtype=float,
        )
        store.set("FAPB", fapb)
        store.set("FMB", fmb)

    def terminate(self, vehicle, ctx):
        pass

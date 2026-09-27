import numpy as np

from cadac.kernel.state import Field


class Hyper6Forces:
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
        cx = store.get("cx")
        cy = store.get("cy")
        cz = store.get("cz")
        cll = store.get("cll")
        clm = store.get("clm")
        cln = store.get("cln")
        farcs = store.get("FARCS") if "FARCS" in store else np.zeros(3)
        fmrcs = store.get("FMRCS") if "FMRCS" in store else np.zeros(3)
        fapb = np.array(
            [
                pdynmc * refa * cx + thrust,
                pdynmc * refa * cy,
                pdynmc * refa * cz,
            ],
            dtype=float,
        )
        fmb = np.array(
            [
                pdynmc * refa * refb * cll,
                pdynmc * refa * refc * clm,
                pdynmc * refa * refb * cln,
            ],
            dtype=float,
        )
        store.set("FAPB", fapb + farcs)
        store.set("FMB", fmb + fmrcs)

    def terminate(self, vehicle, ctx):
        pass

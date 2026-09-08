import numpy as np

from cadac.kernel.state import Field


class Sam6Forces:
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
        mtvc = store.get("mtvc") if "mtvc" in store else 0
        if mtvc != 0:
            raise ValueError(f"mtvc={mtvc!r} not supported in this slice")
        pdynmc = store.get("pdynmc")
        thrust = store.get("thrust")
        refl = store.get("refl")
        refa = store.get("refa")
        ca = store.get("ca")
        cy = store.get("cy")
        cn = store.get("cn")
        cll = store.get("cll")
        clm = store.get("clm")
        cln = store.get("cln")
        farcs = store.get("FARCS") if "FARCS" in store else np.zeros(3)
        fmrcs = store.get("FMRCS") if "FMRCS" in store else np.zeros(3)
        fapb = np.array(
            [
                -pdynmc * refa * ca + thrust,
                pdynmc * refa * cy,
                -pdynmc * refa * cn,
            ],
            dtype=float,
        )
        fmb = np.array(
            [
                pdynmc * refa * refl * cll,
                pdynmc * refa * refl * clm,
                pdynmc * refa * refl * cln,
            ],
            dtype=float,
        )
        store.set("FAPB", fapb + farcs)
        store.set("FMB", fmb + fmrcs)

    def terminate(self, vehicle, ctx):
        pass

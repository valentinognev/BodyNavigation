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
        fpb = store.get("FPB") if "FPB" in store else np.zeros(3)
        fmpb = store.get("FMPB") if "FMPB" in store else np.zeros(3)

        fapb = np.array(
            [
                -pdynmc * refa * ca,
                pdynmc * refa * cy,
                -pdynmc * refa * cn,
            ],
            dtype=float,
        )
        if mtvc == 0:
            fapb[0] = fapb[0] + thrust
        else:
            fapb = fapb + fpb

        fapb = fapb + farcs

        fmb = np.array(
            [
                pdynmc * refa * refl * cll,
                pdynmc * refa * refl * clm,
                pdynmc * refa * refl * cln,
            ],
            dtype=float,
        )
        if mtvc != 0:
            fmb = fmb + fmpb
        fmb = fmb + fmrcs

        store.set("FAPB", fapb)
        store.set("FMB", fmb)

    def terminate(self, vehicle, ctx):
        pass

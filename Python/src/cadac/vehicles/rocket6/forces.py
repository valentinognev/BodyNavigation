import numpy as np

from cadac.kernel.state import Field


class Rocket6Forces:
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
        mprop = store.get("mprop")
        thrust = store.get("thrust")
        mrcs_moment = store.get("mrcs_moment")
        mrcs_force = store.get("mrcs_force")
        refa = store.get("refa")
        refd = store.get("refd")
        cy = store.get("cy")
        cll = store.get("cll")
        clm = store.get("clm")
        cln = store.get("cln")
        cx = store.get("cx")
        cz = store.get("cz")
        mtvc = store.get("mtvc")
        names = store.names()
        farcs = store.get("FARCS") if "FARCS" in names else np.zeros(3)
        fmrcs = store.get("FMRCS") if "FMRCS" in names else np.zeros(3)
        fpb = store.get("FPB") if "FPB" in names else np.zeros(3)
        fmpb = store.get("FMPB") if "FMPB" in names else np.zeros(3)

        fapb = np.array(
            [pdynmc * refa * cx, pdynmc * refa * cy, pdynmc * refa * cz],
            dtype=float,
        )
        fmb = np.array(
            [
                pdynmc * refa * refd * cll,
                pdynmc * refa * refd * clm,
                pdynmc * refa * refd * cln,
            ],
            dtype=float,
        )

        if mtvc == 1 or mtvc == 2 or mtvc == 3:
            fapb = fapb + fpb
            fmb = fmb + fmpb
        elif mprop:
            fapb[0] = fapb[0] + thrust

        if mrcs_force == 1 or mrcs_force == 2:
            fapb = fapb + farcs

        if mrcs_moment > 0 and mrcs_moment <= 23:
            fmb = fmb + fmrcs

        store.set("FAPB", fapb)
        store.set("FMB", fmb)

    def terminate(self, vehicle, ctx):
        pass

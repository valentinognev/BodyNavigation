from cadac.constants import DEG, RAD
from cadac.kernel.state import Field


class Sraam6Control:
    name = "control"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("maut", 0, "int", "data", "control"),
            Field("mfreeze", 0, "int", "data", "control"),
            Field("wacl", 0.0, "real", "data", "control"),
            Field("zacl", 0.0, "real", "data", "control"),
            Field("pacl", 0.0, "real", "data", "control"),
            Field("alimit", 0.0, "real", "data", "control"),
            Field("dqlimx", 0.0, "real", "data", "control"),
            Field("drlimx", 0.0, "real", "data", "control"),
            Field("dplimx", 0.0, "real", "data", "control"),
            Field("phicomx", 0.0, "real", "data", "control"),
            Field("wrcl", 0.0, "real", "data", "control"),
            Field("zrcl", 0.0, "real", "data", "control"),
            Field("yyd", 0.0, "real", "state", "control"),
            Field("yy", 0.0, "real", "state", "control"),
            Field("zzd", 0.0, "real", "state", "control"),
            Field("zz", 0.0, "real", "state", "control"),
            Field("dpcx", 0.0, "real", "out", "control"),
            Field("dqcx", 0.0, "real", "out", "control"),
            Field("drcx", 0.0, "real", "out", "control"),
            Field("GAINFB", zeros3, "vec", "diag", "control"),
            Field("gainp", 0.0, "real", "data", "control"),
            Field("gkp", 0.0, "real", "diag", "control"),
            Field("gkphi", 0.0, "real", "diag", "control"),
            Field("fspb2m", 0.0, "real", "diag", "control"),
            Field("fspb2mt", 0.0, "real", "diag", "control"),
            Field("fspb3m", 0.0, "real", "diag", "control"),
            Field("fspb3mt", 0.0, "real", "diag", "control"),
            Field("qqxm", 0.0, "real", "diag", "control"),
            Field("qqxmt", 0.0, "real", "diag", "control"),
            Field("rrxm", 0.0, "real", "diag", "control"),
            Field("rrxmt", 0.0, "real", "diag", "control"),
            Field("dqcxm", 0.0, "real", "diag", "control"),
            Field("dqcxmt", 0.0, "real", "diag", "control"),
            Field("drcxm", 0.0, "real", "diag", "control"),
            Field("drcxmt", 0.0, "real", "diag", "control"),
            Field("isetc2", 0.0, "real", "init", "control"),
            Field("factwacl", 0.0, "real", "data", "control"),
            Field("factzacl", 0.0, "real", "data", "control"),
            Field("zetlagr", 0.0, "real", "data", "control"),
            Field("ratelimx", 0.0, "real", "data", "control"),
            Field("zrate", 0.0, "real", "diag", "control"),
            Field("grate", 0.0, "real", "diag", "control"),
            Field("wnlagr", 0.0, "real", "diag", "control"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        pass

    def control_roll(self, vehicle):
        store = vehicle.store
        phicomx = store.get("phicomx")
        wrcl = store.get("wrcl")
        zrcl = store.get("zrcl")
        phiblx = store.get("phiblx")
        dlp = store.get("dlp")
        dld = store.get("dld")
        pp = store.get("pp")
        gkp = (2.0 * zrcl * wrcl + dlp) / dld
        gkphi = wrcl * wrcl / dld
        ephi = gkphi * (phicomx - phiblx) * RAD
        dpc = ephi - gkp * pp
        dpcx = dpc * DEG
        store.set("dpcx", dpcx)
        store.set("gkp", gkp)
        store.set("gkphi", gkphi)

    def terminate(self, vehicle, ctx):
        pass

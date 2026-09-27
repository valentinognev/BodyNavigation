from math import cos, sin

from cadac.constants import DEG, RAD
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field


def _sign(variable):
    if variable < 0:
        return -1
    return 1


class Rocket6Tvc:
    name = "tvc"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        plot = ("plot",)
        for field in (
            Field("mtvc", 0, "int", "data", "tvc"),
            Field("tvclimx", 0.0, "real", "data", "tvc"),
            Field("dtvclimx", 0.0, "real", "data", "tvc"),
            Field("wntvc", 0.0, "real", "data", "tvc"),
            Field("zettvc", 0.0, "real", "data", "tvc"),
            Field("factgtvc", 0.0, "real", "data", "tvc"),
            Field("gtvc", 0.0, "real", "data", "tvc"),
            Field("parm", 0.0, "real", "data", "tvc"),
            Field("FPB", zeros3, "vec", "out", "tvc"),
            Field("FMPB", zeros3, "vec", "out", "tvc"),
            Field("etax", 0.0, "real", "diag", "tvc", plot),
            Field("zetx", 0.0, "real", "diag", "tvc", plot),
            Field("etacx", 0.0, "real", "diag", "tvc"),
            Field("zetcx", 0.0, "real", "diag", "tvc"),
            Field("etasd", 0.0, "real", "state", "tvc"),
            Field("zetad", 0.0, "real", "state", "tvc"),
            Field("etas", 0.0, "real", "state", "tvc"),
            Field("zeta", 0.0, "real", "state", "tvc"),
            Field("detasd", 0.0, "real", "state", "tvc"),
            Field("dzetad", 0.0, "real", "state", "tvc"),
            Field("detas", 0.0, "real", "state", "tvc"),
            Field("dzeta", 0.0, "real", "state", "tvc"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mtvc = store.get("mtvc")
        if mtvc == 0:
            return
        if mtvc != 2:
            raise ValueError(f"unknown mtvc {mtvc}")
        dt = ctx.int_step
        gtvc = store.get("gtvc")
        parm = store.get("parm")
        xcg = store.get("xcg")
        thrust = store.get("thrust")
        etac = gtvc * store.get("delecx") * RAD
        zetc = gtvc * store.get("delrcx") * RAD
        eta, zet = self._tvc_scnd(vehicle, etac, zetc, dt)
        fpb0 = cos(eta) * cos(zet) * thrust
        fpb1 = cos(eta) * sin(zet) * thrust
        fpb2 = -sin(eta) * thrust
        arm = parm - xcg
        store.set("FPB", (fpb0, fpb1, fpb2))
        store.set("FMPB", (0.0, arm * fpb2, -arm * fpb1))
        store.set("etax", eta * DEG)
        store.set("zetx", zet * DEG)
        store.set("etacx", etac * DEG)
        store.set("zetcx", zetc * DEG)

    def _tvc_scnd(self, vehicle, etac, zetc, int_step):
        store = vehicle.store
        tvclimx = store.get("tvclimx")
        dtvclimx = store.get("dtvclimx")
        wntvc = store.get("wntvc")
        zettvc = store.get("zettvc")
        etasd = store.get("etasd")
        zetad = store.get("zetad")
        etas = store.get("etas")
        zeta = store.get("zeta")
        detasd = store.get("detasd")
        dzetad = store.get("dzetad")
        detas = store.get("detas")
        dzeta = store.get("dzeta")

        if abs(etas) > tvclimx * RAD:
            etas = tvclimx * RAD * _sign(etas)
            if etas * detas > 0.0:
                detas = 0.0
        iflag = 0
        if abs(detas) > dtvclimx * RAD:
            iflag = 1
            detas = dtvclimx * RAD * _sign(detas)
        etasd_new = detas
        etas = integrate(etasd_new, etasd, etas, int_step)
        etasd = etasd_new
        eetas = etac - etas
        detasd_new = wntvc * wntvc * eetas - 2.0 * zettvc * wntvc * etasd
        detas = integrate(detasd_new, detasd, detas, int_step)
        detasd = detasd_new
        if iflag and detas * detasd > 0.0:
            detasd = 0.0
        eta = etas

        if abs(zeta) > tvclimx * RAD:
            zeta = tvclimx * RAD * _sign(zeta)
            if zeta * dzeta > 0.0:
                dzeta = 0.0
        iflag = 0
        if abs(dzeta) > dtvclimx * RAD:
            iflag = 1
            dzeta = dtvclimx * RAD * _sign(dzeta)
        zetad_new = dzeta
        zeta = integrate(zetad_new, zetad, zeta, int_step)
        zetad = zetad_new
        ezeta = zetc - zeta
        dzetad_new = wntvc * wntvc * ezeta - 2.0 * zettvc * wntvc * zetad
        dzeta = integrate(dzetad_new, dzetad, dzeta, int_step)
        dzetad = dzetad_new
        if iflag and dzeta * dzetad > 0.0:
            dzetad = 0.0
        zet = zeta

        store.set("etasd", etasd)
        store.set("zetad", zetad)
        store.set("etas", etas)
        store.set("zeta", zeta)
        store.set("detasd", detasd)
        store.set("dzetad", dzetad)
        store.set("detas", detas)
        store.set("dzeta", dzeta)
        return eta, zet

    def terminate(self, vehicle, ctx):
        pass

from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field


def _sign(variable):
    if variable < 0:
        return -1
    return 1


class Agm6Actuator:
    name = "actuator"

    def define(self, vehicle):
        store = vehicle.store
        plot = ("plot",)
        for field in (
            Field("mact", 0, "int", "data", "actuator"),
            Field("dlimx", 0.0, "real", "data", "actuator"),
            Field("ddlimx", 0.0, "real", "data", "actuator"),
            Field("wnact", 0.0, "real", "data", "actuator"),
            Field("zetact", 0.0, "real", "data", "actuator"),
            Field("dpx", 0.0, "real", "out", "actuator", plot),
            Field("dqx", 0.0, "real", "out", "actuator", plot),
            Field("drx", 0.0, "real", "out", "actuator", plot),
            Field("delx1", 0.0, "real", "diag", "actuator"),
            Field("delx2", 0.0, "real", "diag", "actuator"),
            Field("delx3", 0.0, "real", "diag", "actuator"),
            Field("delx4", 0.0, "real", "diag", "actuator"),
            Field("dxd1", 0.0, "real", "state", "actuator"),
            Field("dxd2", 0.0, "real", "state", "actuator"),
            Field("dxd3", 0.0, "real", "state", "actuator"),
            Field("dxd4", 0.0, "real", "state", "actuator"),
            Field("dx1", 0.0, "real", "state", "actuator"),
            Field("dx2", 0.0, "real", "state", "actuator"),
            Field("dx3", 0.0, "real", "state", "actuator"),
            Field("dx4", 0.0, "real", "state", "actuator"),
            Field("ddxd1", 0.0, "real", "state", "actuator"),
            Field("ddxd2", 0.0, "real", "state", "actuator"),
            Field("ddxd3", 0.0, "real", "state", "actuator"),
            Field("ddxd4", 0.0, "real", "state", "actuator"),
            Field("ddx1", 0.0, "real", "state", "actuator"),
            Field("ddx2", 0.0, "real", "state", "actuator"),
            Field("ddx3", 0.0, "real", "state", "actuator"),
            Field("ddx4", 0.0, "real", "state", "actuator"),
            Field("delcx1", 0.0, "real", "diag", "actuator"),
            Field("delcx2", 0.0, "real", "diag", "actuator"),
            Field("delcx3", 0.0, "real", "diag", "actuator"),
            Field("delcx4", 0.0, "real", "diag", "actuator"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        dt = ctx.int_step
        mact = store.get("mact")
        dlimx = store.get("dlimx")
        dpcx = store.get("dpcx")
        dqcx = store.get("dqcx")
        drcx = store.get("drcx")
        delcx1 = -dpcx + dqcx - drcx
        delcx2 = -dpcx + dqcx + drcx
        delcx3 = +dpcx + dqcx - drcx
        delcx4 = +dpcx + dqcx + drcx
        if mact < 2:
            delx1 = delcx1
            if abs(delx1) > dlimx:
                delx1 = dlimx * _sign(delx1)
            delx2 = delcx2
            if abs(delx2) > dlimx:
                delx2 = dlimx * _sign(delx2)
            delx3 = delcx3
            if abs(delx3) > dlimx:
                delx3 = dlimx * _sign(delx3)
            delx4 = delcx4
            if abs(delx4) > dlimx:
                delx4 = dlimx * _sign(delx4)
            dpx = (-delx1 - delx2 + delx3 + delx4) / 4.0
            dqx = (+delx1 + delx2 + delx3 + delx4) / 4.0
            drx = (-delx1 + delx2 - delx3 + delx4) / 4.0
            store.set("dpx", dpx)
            store.set("dqx", dqx)
            store.set("drx", drx)
            store.set("delx1", delx1)
            store.set("delx2", delx2)
            store.set("delx3", delx3)
            store.set("delx4", delx4)
            store.set("delcx1", delcx1)
            store.set("delcx2", delcx2)
            store.set("delcx3", delcx3)
            store.set("delcx4", delcx4)
        elif mact == 2:
            self._actuator_scnd(vehicle, dt)
        else:
            raise ValueError(f"unknown mact {mact}")

    def _actuator_scnd(self, vehicle, int_step):
        store = vehicle.store
        dlimx = store.get("dlimx")
        ddlimx = store.get("ddlimx")
        wnact = store.get("wnact")
        zetact = store.get("zetact")
        dpcx = store.get("dpcx")
        dqcx = store.get("dqcx")
        drcx = store.get("drcx")
        dxd1 = store.get("dxd1")
        dxd2 = store.get("dxd2")
        dxd3 = store.get("dxd3")
        dxd4 = store.get("dxd4")
        dx1 = store.get("dx1")
        dx2 = store.get("dx2")
        dx3 = store.get("dx3")
        dx4 = store.get("dx4")
        ddxd1 = store.get("ddxd1")
        ddxd2 = store.get("ddxd2")
        ddxd3 = store.get("ddxd3")
        ddxd4 = store.get("ddxd4")
        ddx1 = store.get("ddx1")
        ddx2 = store.get("ddx2")
        ddx3 = store.get("ddx3")
        ddx4 = store.get("ddx4")
        delcx1 = -dpcx + dqcx - drcx
        delcx2 = -dpcx + dqcx + drcx
        delcx3 = +dpcx + dqcx - drcx
        delcx4 = +dpcx + dqcx + drcx
        dxd1, dx1, ddxd1, ddx1 = _fin_scnd(
            delcx1, dxd1, dx1, ddxd1, ddx1, dlimx, ddlimx, wnact, zetact, int_step
        )
        dxd2, dx2, ddxd2, ddx2 = _fin_scnd(
            delcx2, dxd2, dx2, ddxd2, ddx2, dlimx, ddlimx, wnact, zetact, int_step
        )
        dxd3, dx3, ddxd3, ddx3 = _fin_scnd(
            delcx3, dxd3, dx3, ddxd3, ddx3, dlimx, ddlimx, wnact, zetact, int_step
        )
        dxd4, dx4, ddxd4, ddx4 = _fin_scnd(
            delcx4, dxd4, dx4, ddxd4, ddx4, dlimx, ddlimx, wnact, zetact, int_step
        )
        delx1 = dx1
        delx2 = dx2
        delx3 = dx3
        delx4 = dx4
        dpx = (-delx1 - delx2 + delx3 + delx4) / 4.0
        dqx = (+delx1 + delx2 + delx3 + delx4) / 4.0
        drx = (-delx1 + delx2 - delx3 + delx4) / 4.0
        store.set("dxd1", dxd1)
        store.set("dxd2", dxd2)
        store.set("dxd3", dxd3)
        store.set("dxd4", dxd4)
        store.set("dx1", dx1)
        store.set("dx2", dx2)
        store.set("dx3", dx3)
        store.set("dx4", dx4)
        store.set("ddxd1", ddxd1)
        store.set("ddxd2", ddxd2)
        store.set("ddxd3", ddxd3)
        store.set("ddxd4", ddxd4)
        store.set("ddx1", ddx1)
        store.set("ddx2", ddx2)
        store.set("ddx3", ddx3)
        store.set("ddx4", ddx4)
        store.set("dpx", dpx)
        store.set("dqx", dqx)
        store.set("drx", drx)
        store.set("delx1", delx1)
        store.set("delx2", delx2)
        store.set("delx3", delx3)
        store.set("delx4", delx4)
        store.set("delcx1", delcx1)
        store.set("delcx2", delcx2)
        store.set("delcx3", delcx3)
        store.set("delcx4", delcx4)

    def terminate(self, vehicle, ctx):
        pass


def _fin_scnd(delcx, dxd, dx, ddxd, ddx, dlimx, ddlimx, wnact, zetact, int_step):
    if abs(dx) > dlimx:
        dx = dlimx * _sign(dx)
        if dx * ddx > 0:
            ddx = 0.0
    iflag = 0
    if abs(ddx) > ddlimx:
        iflag = 1
        ddx = ddlimx * _sign(ddx)
    dxd_new = ddx
    dx = integrate(dxd_new, dxd, dx, int_step)
    dxd = dxd_new
    edx = delcx - dx
    ddxd_new = wnact * wnact * edx - 2.0 * zetact * wnact * dxd
    ddx = integrate(ddxd_new, ddxd, ddx, int_step)
    ddxd = ddxd_new
    if iflag and ddx * ddxd > 0:
        ddxd = 0.0
    return dxd, dx, ddxd, ddx

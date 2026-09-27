from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field


def _sign(variable):
    if variable < 0:
        return -1
    return 1


class Sam6Actuator:
    name = "actuator"

    def define(self, vehicle):
        store = vehicle.store
        plot = ("plot",)
        scrn_plot = ("scrn", "plot")
        for field in (
            Field("mact", 0, "int", "data", "actuator"),
            Field("dlimx", 0.0, "real", "data", "actuator"),
            Field("ddlimx", 0.0, "real", "data", "actuator"),
            Field("wnact", 0.0, "real", "data", "actuator"),
            Field("zetact", 0.0, "real", "data", "actuator"),
            Field("dpx", 0.0, "real", "out", "actuator", plot),
            Field("dqx", 0.0, "real", "out", "actuator", plot),
            Field("drx", 0.0, "real", "out", "actuator", plot),
            Field("delx1", 0.0, "real", "diag", "actuator", scrn_plot),
            Field("delx2", 0.0, "real", "diag", "actuator", scrn_plot),
            Field("delx3", 0.0, "real", "diag", "actuator", scrn_plot),
            Field("delx4", 0.0, "real", "diag", "actuator", scrn_plot),
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
        mact = store.get("mact")
        dlimx = store.get("dlimx")
        dpcx = store.get("dpcx")
        dqcx = store.get("dqcx")
        drcx = store.get("drcx")
        delcx1 = -dpcx - drcx
        delcx2 = -dpcx + dqcx
        delcx3 = -dpcx + drcx
        delcx4 = -dpcx - dqcx
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
            dpx = 0.25 * (-delx1 - delx2 - delx3 - delx4)
            dqx = 0.5 * (delx2 - delx4)
            drx = 0.5 * (-delx1 + delx3)
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
            self._actuator_scnd(vehicle, ctx.int_step)
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
        delcx = (
            -dpcx - drcx,
            -dpcx + dqcx,
            -dpcx + drcx,
            -dpcx - dqcx,
        )
        dxd = [store.get(f"dxd{i}") for i in range(1, 5)]
        dx = [store.get(f"dx{i}") for i in range(1, 5)]
        ddxd = [store.get(f"ddxd{i}") for i in range(1, 5)]
        ddx = [store.get(f"ddx{i}") for i in range(1, 5)]
        for i in range(4):
            if abs(dx[i]) > dlimx:
                dx[i] = dlimx * _sign(dx[i])
                if dx[i] * ddx[i] > 0:
                    ddx[i] = 0.0
            iflag = 0
            if abs(ddx[i]) > ddlimx:
                iflag = 1
                ddx[i] = ddlimx * _sign(ddx[i])
            dxd_new = ddx[i]
            dx[i] = integrate(dxd_new, dxd[i], dx[i], int_step)
            dxd[i] = dxd_new
            edx = delcx[i] - dx[i]
            ddxd_new = wnact * wnact * edx - 2.0 * zetact * wnact * dxd[i]
            ddx[i] = integrate(ddxd_new, ddxd[i], ddx[i], int_step)
            ddxd[i] = ddxd_new
            if iflag and ddx[i] * ddxd[i] > 0:
                ddxd[i] = 0.0
        delx1, delx2, delx3, delx4 = dx
        dpx = 0.25 * (-delx1 - delx2 - delx3 - delx4)
        dqx = 0.5 * (delx2 - delx4)
        drx = 0.5 * (-delx1 + delx3)
        for i in range(4):
            n = i + 1
            store.set(f"dxd{n}", dxd[i])
            store.set(f"dx{n}", dx[i])
            store.set(f"ddxd{n}", ddxd[i])
            store.set(f"ddx{n}", ddx[i])
        store.set("dpx", dpx)
        store.set("dqx", dqx)
        store.set("drx", drx)
        store.set("delx1", delx1)
        store.set("delx2", delx2)
        store.set("delx3", delx3)
        store.set("delx4", delx4)
        store.set("delcx1", delcx[0])
        store.set("delcx2", delcx[1])
        store.set("delcx3", delcx[2])
        store.set("delcx4", delcx[3])

    def terminate(self, vehicle, ctx):
        pass

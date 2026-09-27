import numpy as np

from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field


def _sign(variable):
    if variable < 0:
        return -1
    return 1


class Hyper6Actuator:
    name = "actuator"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        plot = ("plot",)
        scrn_plot = ("scrn", "plot")
        for field in (
            Field("mact", 0, "int", "data", "actuator"),
            Field("dlimx", 0.0, "real", "data", "actuator"),
            Field("ddlimx", 0.0, "real", "data", "actuator"),
            Field("wnact", 0.0, "real", "data", "actuator"),
            Field("zetact", 0.0, "real", "data", "actuator"),
            Field("delax", 0.0, "real", "out", "actuator", scrn_plot),
            Field("delex", 0.0, "real", "out", "actuator", scrn_plot),
            Field("delrx", 0.0, "real", "out", "actuator", scrn_plot),
            Field("elvlx", 0.0, "real", "dia", "actuator", plot),
            Field("elvrx", 0.0, "real", "dia", "actuator", plot),
            Field("elvlcx", 0.0, "real", "dia", "actuator"),
            Field("elvrcx", 0.0, "real", "dia", "actuator"),
            Field("DXD", zeros3, "vec", "state", "actuator"),
            Field("DX", zeros3, "vec", "state", "actuator"),
            Field("DDXD", zeros3, "vec", "state", "actuator"),
            Field("DDX", zeros3, "vec", "state", "actuator"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        dt = ctx.int_step
        mact = store.get("mact")
        dlimx = store.get("dlimx")
        delacx = store.get("delacx")
        delecx = store.get("delecx")
        delrcx = store.get("delrcx")
        elvlcx = delecx + delacx
        elvrcx = delecx - delacx
        actcx = np.array([elvlcx, elvrcx, delrcx], dtype=float)
        if mact == 0:
            actx = actcx.copy()
            for i in range(3):
                if abs(actx[i]) > dlimx:
                    actx[i] = dlimx * _sign(actx[i])
        elif mact == 2:
            actx = self._actuator_scnd(vehicle, actcx, dt)
        else:
            raise ValueError(f"unknown mact {mact}")
        elvlx = float(actx[0])
        elvrx = float(actx[1])
        delrx = float(actx[2])
        delax = (elvlx - elvrx) / 2.0
        delex = (elvlx + elvrx) / 2.0
        store.set("delax", delax)
        store.set("delex", delex)
        store.set("delrx", delrx)
        store.set("elvlx", elvlx)
        store.set("elvrx", elvrx)
        store.set("elvlcx", elvlcx)
        store.set("elvrcx", elvrcx)

    def _actuator_scnd(self, vehicle, actcx, int_step):
        store = vehicle.store
        dlimx = store.get("dlimx")
        ddlimx = store.get("ddlimx")
        wnact = store.get("wnact")
        zetact = store.get("zetact")
        dxd = np.asarray(store.get("DXD"), dtype=float).copy()
        dx = np.asarray(store.get("DX"), dtype=float).copy()
        ddxd = np.asarray(store.get("DDXD"), dtype=float).copy()
        ddx = np.asarray(store.get("DDX"), dtype=float).copy()
        for i in range(3):
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
            edx = actcx[i] - dx[i]
            ddxd_new = wnact * wnact * edx - 2.0 * zetact * wnact * dxd[i]
            ddx[i] = integrate(ddxd_new, ddxd[i], ddx[i], int_step)
            ddxd[i] = ddxd_new
            if iflag and ddx[i] * ddxd[i] > 0:
                ddxd[i] = 0.0
        store.set("DXD", dxd)
        store.set("DX", dx)
        store.set("DDXD", ddxd)
        store.set("DDX", ddx)
        return dx

    def terminate(self, vehicle, ctx):
        pass

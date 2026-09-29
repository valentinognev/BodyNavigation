import numpy as np

from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field


def _sign(variable):
    if variable < 0:
        return -1
    return 1


class Rocket6Actuator:
    name = "actuator"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        plot = ("plot",)
        scrn_plot = ("scrn", "plot")
        for field in (
            Field("mact", 0, "int", "data", "actuator"),
            Field("num_fins", 0, "int", "data", "actuator"),
            Field("dlimx", 0.0, "real", "data", "actuator"),
            Field("dlimx_min", 0.0, "real", "data", "actuator"),
            Field("ddlimx", 0.0, "real", "data", "actuator"),
            Field("wnact", 0.0, "real", "data", "actuator", plot),
            Field("zetact", 0.0, "real", "data", "actuator"),
            Field("mvehicle", 0, "int", "out", "actuator"),
            Field("factwnact", 0.0, "real", "data", "actuator"),
            Field("wnact_limit", 0.0, "real", "data", "actuator"),
            Field("delax", 0.0, "real", "out", "actuator", scrn_plot),
            Field("delex", 0.0, "real", "out", "actuator", scrn_plot),
            Field("delrx", 0.0, "real", "out", "actuator", scrn_plot),
            Field("elvlx", 0.0, "real", "diag", "actuator"),
            Field("elvrx", 0.0, "real", "diag", "actuator"),
            Field("elvlcx", 0.0, "real", "diag", "actuator"),
            Field("elvrcx", 0.0, "real", "diag", "actuator"),
            Field("DXD", zeros3, "vec", "state", "actuator"),
            Field("DX", zeros3, "vec", "state", "actuator"),
            Field("DDXD", zeros3, "vec", "state", "actuator"),
            Field("DDX", zeros3, "vec", "state", "actuator"),
            Field("DYD", zeros3, "vec", "state", "actuator"),
            Field("DY", zeros3, "vec", "state", "actuator"),
            Field("DDYD", zeros3, "vec", "state", "actuator"),
            Field("DDY", zeros3, "vec", "state", "actuator"),
            Field("delx1", 0.0, "real", "out", "actuator"),
            Field("delx2", 0.0, "real", "out", "actuator"),
            Field("delx3", 0.0, "real", "out", "actuator"),
            Field("delx4", 0.0, "real", "out", "actuator"),
            Field("delx5", 0.0, "real", "out", "actuator"),
            Field("delx6", 0.0, "real", "out", "actuator"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        dt = ctx.int_step
        mact = store.get("mact")
        dlimx = store.get("dlimx")
        dlimx_min = store.get("dlimx_min")
        delacx = store.get_optional("delacx", 0.0)
        delecx = store.get_optional("delecx", 0.0)
        delrcx = store.get_optional("delrcx", 0.0)

        morder = mact // 10
        mvehicle = mact % 10

        delax = 0.0
        delex = 0.0
        delrx = 0.0
        elvlx = 0.0
        elvrx = 0.0
        elvlcx = 0.0
        elvrcx = 0.0
        delx1 = 0.0
        delx2 = 0.0
        delx3 = 0.0
        delx4 = 0.0
        delx5 = 0.0
        delx6 = 0.0

        if mvehicle == 2:
            num_fins = 4
            dlimx_min = -dlimx
            # Rocket four-fin mix (old C++ ROCKET6 actuator.cpp)
            delcx1 = -delacx + delecx - delrcx
            delcx2 = -delacx + delecx + delrcx
            delcx3 = +delacx + delecx - delrcx
            delcx4 = +delacx + delecx + delrcx
            actcz = np.array(
                [delcx1, delcx2, delcx3, delcx4, 0.0, 0.0], dtype=float
            )
            if morder == 0:
                actz = self._actuator_0th(actcz, dlimx, dlimx_min, num_fins)
            elif morder == 2:
                actz = self._actuator_scnd(
                    vehicle, actcz, dlimx, dlimx_min, num_fins, dt
                )
            else:
                actz = np.zeros(6, dtype=float)
            delx1 = float(actz[0])
            delx2 = float(actz[1])
            delx3 = float(actz[2])
            delx4 = float(actz[3])
            delax = (-delx1 - delx2 + delx3 + delx4) / 4.0
            delex = (+delx1 + delx2 + delx3 + delx4) / 4.0
            delrx = (-delx1 + delx2 - delx3 + delx4) / 4.0

        store.set("mvehicle", mvehicle)
        store.set("delax", delax)
        store.set("delex", delex)
        store.set("delrx", delrx)
        store.set("elvlx", elvlx)
        store.set("elvrx", elvrx)
        store.set("elvlcx", elvlcx)
        store.set("elvrcx", elvrcx)
        store.set("delx1", delx1)
        store.set("delx2", delx2)
        store.set("delx3", delx3)
        store.set("delx4", delx4)
        store.set("delx5", delx5)
        store.set("delx6", delx6)

    def _actuator_0th(self, actcz, dlimx, dlimx_min, num_fins):
        actz = np.asarray(actcz, dtype=float).copy()
        for i in range(num_fins):
            if actz[i] > dlimx:
                actz[i] = dlimx
            if actz[i] < dlimx_min:
                actz[i] = dlimx_min
        return actz

    def _actuator_scnd(self, vehicle, actcz, dlimx, dlimx_min, num_fins, int_step):
        store = vehicle.store
        ddlimx = store.get("ddlimx")
        wnact = store.get("wnact")
        zetact = store.get("zetact")
        dxd = np.asarray(store.get("DXD"), dtype=float).copy()
        dx = np.asarray(store.get("DX"), dtype=float).copy()
        ddxd = np.asarray(store.get("DDXD"), dtype=float).copy()
        ddx = np.asarray(store.get("DDX"), dtype=float).copy()
        dyd = np.asarray(store.get("DYD"), dtype=float).copy()
        dy = np.asarray(store.get("DY"), dtype=float).copy()
        ddyd = np.asarray(store.get("DDYD"), dtype=float).copy()
        ddy = np.asarray(store.get("DDY"), dtype=float).copy()

        dzd = np.zeros(6, dtype=float)
        dz = np.zeros(6, dtype=float)
        ddzd = np.zeros(6, dtype=float)
        ddz = np.zeros(6, dtype=float)
        for n in range(3):
            dzd[n] = dxd[n]
            dzd[n + 3] = dyd[n]
            dz[n] = dx[n]
            dz[n + 3] = dy[n]
            ddzd[n] = ddxd[n]
            ddzd[n + 3] = ddyd[n]
            ddz[n] = ddx[n]
            ddz[n + 3] = ddy[n]

        for i in range(num_fins):
            if dz[i] > dlimx:
                dz[i] = dlimx
                if dz[i] * ddz[i] > 0:
                    ddz[i] = 0.0
            if dz[i] < dlimx_min:
                dz[i] = dlimx_min
                if dz[i] * ddz[i] > 0:
                    ddz[i] = 0.0
            iflag = 0
            if abs(ddz[i]) > ddlimx:
                iflag = 1
                ddz[i] = ddlimx * _sign(ddz[i])
            dzd_new = ddz[i]
            dz[i] = integrate(dzd_new, dzd[i], dz[i], int_step)
            dzd[i] = dzd_new
            edx = actcz[i] - dz[i]
            ddzd_new = wnact * wnact * edx - 2.0 * zetact * wnact * dzd[i]
            ddz[i] = integrate(ddzd_new, ddzd[i], ddz[i], int_step)
            ddzd[i] = ddzd_new

            if dz[i] > dlimx:
                dz[i] = dlimx
                if dz[i] * ddz[i] > 0:
                    ddz[i] = 0.0
            if dz[i] < dlimx_min:
                dz[i] = dlimx_min
                if dz[i] * ddz[i] > 0:
                    ddz[i] = 0.0
            iflag = 0
            if abs(ddz[i]) > ddlimx:
                iflag = 1
                ddz[i] = ddlimx * _sign(ddz[i])
            if iflag and ddz[i] * ddzd[i] > 0:
                ddzd[i] = 0.0

        for n in range(3):
            dxd[n] = dzd[n]
            dyd[n] = dzd[n + 3]
            dx[n] = dz[n]
            dy[n] = dz[n + 3]
            ddxd[n] = ddzd[n]
            ddyd[n] = ddzd[n + 3]
            ddx[n] = ddz[n]
            ddy[n] = ddz[n + 3]

        store.set("DXD", dxd)
        store.set("DX", dx)
        store.set("DDXD", ddxd)
        store.set("DDX", ddx)
        store.set("DYD", dyd)
        store.set("DY", dy)
        store.set("DDYD", ddyd)
        store.set("DDY", ddy)
        store.set("wnact", wnact)
        return dz

    def terminate(self, vehicle, ctx):
        pass

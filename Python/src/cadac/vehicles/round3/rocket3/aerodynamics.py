"""ROCKET3 aerodynamics (Fortran MODULE.FOR A1)."""

from math import cos, sin

from cadac.constants import DEG
from cadac.kernel.state import Field
from cadac.vehicles.round3.rocket3.stubs import StubModule


class Rocket3Aerodynamics(StubModule):
    name = "aerodynamics"
    _fields = (
        Field("maero", 0, "int", "data", "aerodynamics"),
        Field("sref", 0.0, "real", "data", "aerodynamics"),
        Field("vmassi", 0.0, "real", "data", "aerodynamics"),
        Field("alphax", 0.0, "real", "data", "aerodynamics"),
        Field("phimvx", 0.0, "real", "data", "aerodynamics"),
        Field("cd", 0.0, "real", "out", "aerodynamics"),
        Field("cl", 0.0, "real", "out", "aerodynamics"),
        Field("ca", 0.0, "real", "out", "aerodynamics"),
        Field("cn", 0.0, "real", "out", "aerodynamics"),
        Field("clovercd", 0.0, "real", "diag", "aerodynamics"),
    )

    def execute(self, vehicle, ctx):
        store = vehicle.store
        maero = int(store.get("maero"))
        maert = int(maero / 10.0)
        maerv = maero - maert * 10
        alpha = float(store.get("alphax")) / DEG
        calph = cos(alpha)
        salph = sin(alpha)
        vmach = float(store.get("vmach"))
        mprop = int(store.get("mprop"))

        if maert != 1:
            return

        caa = 0.0
        cnn = 0.0
        if maerv == 1:
            if mprop == 2:
                caa = 0.281 + 0.186 * vmach - 0.056 * vmach**2 + 0.00366 * vmach**3
            else:
                caa = 0.346 + 0.183 * vmach - 0.058 * vmach**2 + 0.00382 * vmach**3
            cnn = (5.006 - 0.519 * vmach + 0.031 * vmach**2) * alpha
        elif maerv == 2:
            if mprop == 2:
                caa = 0.236 - 0.043 * vmach + 0.0029 * vmach**2 - 0.00006 * vmach**3
            else:
                caa = 0.327 - 0.067 * vmach + 0.005 * vmach**2 - 0.0001 * vmach**3
            cnn = (1.714 - 0.038 * vmach + 0.0014 * vmach**2) * alpha
        elif maerv == 3:
            caa = 0.02
            cnn = 1.0 * alpha
        else:
            return

        cdd = caa * calph + cnn * salph
        cll = cnn * calph - caa * salph
        cl = cll
        cd = cdd
        store.set("cl", cl)
        store.set("cd", cd)
        store.set("clovercd", cl / cd)
        store.set("cn", cl * calph + cd * salph)
        store.set("ca", cd * calph - cl * salph)

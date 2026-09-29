"""ROCKET3 propulsion A2 — MPROP 0/1/2 fuel flow / mass (Fortran MODULE.FOR)."""

from cadac.constants import AGRAV
from cadac.kernel.state import Field
from cadac.vehicles.round3.rocket3.stubs import StubModule

_PRESS_SL = 101325.0


class Rocket3Propulsion(StubModule):
    name = "propulsion"
    _fields = (
        Field("mprop", 0, "int", "data", "propulsion"),
        Field("fueli", 0.0, "real", "data", "propulsion"),
        Field("fuelr", 0.0, "real", "data", "propulsion"),
        Field("spi", 0.0, "real", "data", "propulsion"),
        Field("aexit", 0.0, "real", "data", "propulsion"),
        Field("thrtl", 0.0, "real", "data", "propulsion"),
        Field("vmassi", 0.0, "real", "data", "propulsion"),
        Field("tstage", 0.0, "real", "save", "propulsion"),
        Field("fmassf", 0.0, "real", "save", "propulsion"),
        Field("fmassfm", 0.0, "real", "save", "propulsion"),
        Field("fmassfd", 0.0, "real", "diag", "propulsion"),
        Field("vmassim", 0.0, "real", "save", "propulsion"),
        Field("vmass", 0.0, "real", "out", "propulsion"),
        Field("fuel", 0.0, "real", "diag", "propulsion"),
        Field("thrustx", 0.0, "real", "out", "propulsion"),
    )

    def initialize(self, vehicle, ctx):
        store = vehicle.store
        vmassi = float(store.get("vmassi"))
        store.set("fmassf", 0.0)
        store.set("vmass", vmassi)
        store.set("fmassfm", 0.0)
        store.set("vmassim", 0.0)
        store.set("fmassfd", 0.0)

    def execute(self, vehicle, ctx):
        store = vehicle.store
        der = float(ctx.int_step)

        mprop = int(store.get("mprop"))
        fueli = float(store.get("fueli"))
        fuelr = float(store.get("fuelr"))
        spi = float(store.get("spi"))
        vmassi = float(store.get("vmassi"))
        thrtl = float(store.get("thrtl"))
        aexit = float(store.get("aexit"))
        press = float(store.get("press"))

        fmassf = float(store.get("fmassf"))
        fmassfm = float(store.get("fmassfm"))
        fmassfd = float(store.get("fmassfd"))
        vmassim = float(store.get("vmassim"))

        # New stage: initialize expended fuel to zero (VMASSIM - VMASSI > 0).
        if (vmassim - vmassi) > 0.0:
            fmassfm = 0.0
            fmassf = 0.0
            fmassfd = 0.0
        vmassim = vmassi

        # Euler on expended fuel (Fortran ICOOR.EQ.0 path; Python runs once per step).
        fmassf = fmassf + fmassfd * der

        thrust = 0.0
        if mprop == 0:
            thrust = 0.0
            fmassfd = 0.0
            vmass = vmassi - fmassfm
            fuel = fueli - fmassfm
        elif mprop == 1:
            thrust = 0.0
            fmassfd = 0.0
            fmassf = 0.0
            vmass = vmassi - fmassfm
            fuel = fueli - fmassfm
        elif mprop == 2:
            flr = fuelr * thrtl
            thrust = flr * spi * AGRAV + (_PRESS_SL - press) * aexit
            fmassfd = flr
            fuel = fueli - fmassf
            vmass = vmassi - fmassf
            fmassfm = fmassf
            if fuel <= 0.0:
                mprop = 1
                thrust = 0.0
                fmassfd = 0.0
                fmassf = 0.0
        else:
            raise ValueError(f"unknown mprop {mprop}")

        thrustx = thrust / 1000.0

        store.set("mprop", mprop)
        store.set("fmassf", fmassf)
        store.set("fmassfm", fmassfm)
        store.set("fmassfd", fmassfd)
        store.set("vmassim", vmassim)
        store.set("vmass", vmass)
        store.set("fuel", fuel)
        store.set("thrustx", thrustx)

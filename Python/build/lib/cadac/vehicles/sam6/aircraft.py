from math import atan2, sqrt

import numpy as np

from cadac.constants import DEG, EPS, RAD
from cadac.kernel.events import EventEngine
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import skew
from cadac.vehicles.sam6.flat3 import (
    Sam6Flat3Environment,
    Sam6Flat3Kinematics,
    Sam6Flat3Newton,
)


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def _first_missile6(ctx):
    combus = ctx.combus or ()
    for packet in combus:
        if packet.type == "MISSILE6":
            return packet
    raise ValueError("no MISSILE6 packet")


class Sam6AircraftGuidance:
    name = "guidance"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("acft_option", 0, "int", "data", "guidance"),
            Field("guid_gain", 0.0, "real", "data", "guidance"),
            Field("ACOML", zeros3, "vec", "out", "guidance"),
            Field("gturn", 0.0, "real", "data", "guidance"),
            Field("man_start", 0.0, "real", "data", "guidance"),
            Field("man_stop", 0.0, "real", "data", "guidance"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        acft_option = store.get("acft_option")
        if acft_option not in (0, 1, 2):
            raise ValueError(f"acft_option={acft_option!r} not supported")
        time = store.get("time")
        grav = store.get("grav")
        man_start = store.get("man_start")
        man_stop = store.get("man_stop")
        in_window = man_start <= time < man_stop
        if acft_option == 0 or not in_window:
            store.set("ACOML", np.array([0.0, 0.0, -grav], dtype=float))
            return
        if acft_option == 1:
            tvl = np.asarray(store.get("TVL"), dtype=float)
            acomv = np.array(
                [0.0, store.get("gturn") * grav, -grav], dtype=float
            )
            store.set("ACOML", tvl.T @ acomv)
            return
        packet = _first_missile6(ctx)
        stel = np.asarray(packet.vars["SBEL"], dtype=float)
        vtel = np.asarray(packet.vars["VBEL"], dtype=float)
        sael = np.asarray(store.get("SAEL"), dtype=float)
        vael = np.asarray(store.get("VAEL"), dtype=float)
        satl = sael - stel
        dab = np.linalg.norm(satl)
        dum = np.linalg.norm(skew(vael) @ vtel)
        gain = store.get("guid_gain") * dum / dab
        uvtel = vtel / np.linalg.norm(vtel)
        uvael = vael / np.linalg.norm(vael)
        epsl = skew(uvael) @ uvtel
        acoml = skew(epsl) @ uvael * gain
        acoml = acoml + np.array([0.0, 0.0, -grav], dtype=float)
        store.set("ACOML", acoml)

    def terminate(self, vehicle, ctx):
        pass


class Sam6AircraftControl:
    name = "control"

    def define(self, vehicle):
        store = vehicle.store
        com = ("com",)
        for field in (
            Field("phiav", 0.0, "real", "state", "control"),
            Field("phiavd", 0.0, "real", "state", "control"),
            Field("tphi", 0.0, "real", "data", "control"),
            Field("philimx", 0.0, "real", "data", "control"),
            Field("phiavx", 0.0, "real", "out", "control", com),
            Field("phiavcx", 0.0, "real", "diag", "control"),
            Field("anx", 0.0, "real", "state", "control", com),
            Field("anxd", 0.0, "real", "state", "control"),
            Field("tanx", 0.0, "real", "data", "control"),
            Field("alplimx", 0.0, "real", "data", "control"),
            Field("ancomx", 0.0, "real", "diag", "control"),
            Field("clalpha", 0.0, "real", "data", "control"),
            Field("wingloading", 0.0, "real", "data", "control"),
            Field("phiavout", 0.0, "real", "out", "control"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        tphi = store.get("tphi")
        philimx = store.get("philimx")
        tanx = store.get("tanx")
        alplimx = store.get("alplimx")
        clalpha = store.get("clalpha")
        wingloading = store.get("wingloading")
        grav = store.get("grav")
        pdynmc = store.get("pdynmc")
        tvl = np.asarray(store.get("TVL"), dtype=float)
        acft_option = store.get("acft_option")
        acoml = np.asarray(store.get("ACOML"), dtype=float)
        phiav = store.get("phiav")
        phiavd = store.get("phiavd")
        anx = store.get("anx")
        anxd = store.get("anxd")

        acomv = tvl @ acoml
        acoma2 = float(acomv[1])
        acoma3 = float(acomv[2])
        if abs(acoma2) < EPS and abs(acoma3) < EPS:
            phiavc = 0.0
        else:
            phiavc = atan2(acoma2, -acoma3)
        phiavcx = phiavc * DEG

        if tphi:
            phiavd_new = (phiavc - phiav) / tphi
            phiav = integrate(phiavd_new, phiavd, phiav, ctx.int_step)
            phiavd = phiavd_new
        else:
            phiav = phiavc

        phiavx = phiav * DEG
        if abs(phiavx) >= philimx:
            phiavx = philimx * _sign(phiavx)
        phiavout = phiavx * RAD

        ancomx = sqrt(acoma2 * acoma2 + acoma3 * acoma3) / grav
        if tanx:
            anxd_new = (ancomx - anx) / tanx
            anx = integrate(anxd_new, anxd, anx, ctx.int_step)
            anxd = anxd_new
        else:
            anx = ancomx
        if acft_option > 0:
            anlimx = pdynmc * clalpha * alplimx / wingloading
            if abs(anx) >= anlimx:
                anx = anlimx * _sign(anx)

        store.set("phiav", phiav)
        store.set("phiavd", phiavd)
        store.set("anx", anx)
        store.set("anxd", anxd)
        store.set("phiavout", phiavout)
        store.set("phiavx", phiavx)
        store.set("phiavcx", phiavcx)
        store.set("ancomx", ancomx)

    def terminate(self, vehicle, ctx):
        pass


class Sam6AircraftForces:
    name = "forces"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("FSPA", (0.0, 0.0, 0.0), "vec", "out", "forces"),
            Field("acc_longx", 0.0, "real", "data", "forces"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        grav = store.get("grav")
        anx = store.get("anx")
        acc_longx = store.get("acc_longx")
        store.set(
            "FSPA",
            np.array([acc_longx * grav, 0.0, -anx * grav], dtype=float),
        )

    def terminate(self, vehicle, ctx):
        pass


class Sam6Aircraft:
    type = "AIRCRAFT3"

    def __init__(self, name, events=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Sam6Flat3Environment(),
            Sam6Flat3Kinematics(),
            Sam6AircraftGuidance(),
            Sam6AircraftControl(),
            Sam6AircraftForces(),
            Sam6Flat3Newton(),
        ]

    def define(self):
        store = self.store
        orig_define = store.define

        def define_skip_if_exists(field):
            if field.name not in store:
                orig_define(field)

        store.define = define_skip_if_exists
        try:
            for module in self.modules:
                module.define(self)
        finally:
            store.define = orig_define
        self.com_names = [
            name
            for name in self.store.names()
            if "com" in self.store.field(name).outputs
        ]

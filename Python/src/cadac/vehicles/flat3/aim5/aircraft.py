from math import atan2, sqrt

import numpy as np

from cadac.constants import DEG, EPS, RAD
from cadac.eom.flat3 import Flat3Environment, Flat3Kinematics
from cadac.kernel.events import EventEngine
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import cadac_sign, skew
from cadac.vehicles.flat3.aim5.vehicle import Aim5Flat3Newton, _define_aim5_vehicle


class Aim5AircraftForces:
    name = "forces"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("FSPV", (0.0, 0.0, 0.0), "vec", "out", "forces"),
            Field("acc_longx", 0.0, "real", "data", "forces"),
        ):
            if field.name not in store:
                store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        acc_longx = store.get("acc_longx")
        grav = store.get("grav")
        anx = store.get("anx")
        store.set("FSPV", np.array([acc_longx * grav, 0.0, -anx * grav]))

    def terminate(self, vehicle, ctx):
        pass


class Aim5AircraftControl:
    name = "control"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("phiav", 0.0, "real", "state", "control"),
            Field("phiavd", 0.0, "real", "state", "control"),
            Field("tphi", 0.0, "real", "data", "control"),
            Field("philimx", 0.0, "real", "data", "control"),
            Field("phiavx", 0.0, "real", "out", "control"),
            Field("phiavcx", 0.0, "real", "diag", "control"),
            Field("anx", 0.0, "real", "state", "control"),
            Field("anxd", 0.0, "real", "state", "control"),
            Field("tanx", 0.0, "real", "data", "control"),
            Field("alplimx", 0.0, "real", "data", "control"),
            Field("ancomx", 0.0, "real", "diag", "control"),
            Field("clalpha", 0.0, "real", "data", "control"),
            Field("wingloading", 0.0, "real", "data", "control"),
            Field("phiavout", 0.0, "real", "out", "control"),
        ):
            if field.name == "phiavout" and field.name in store:
                continue
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
        tvl = store.get("TVL")
        acft_option = store.get("acft_option")
        acoml = store.get("ACOML")
        phiav = store.get("phiav")
        phiavd = store.get("phiavd")
        anx = store.get("anx")
        anxd = store.get("anxd")
        int_step = ctx.int_step

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
            phiav = integrate(phiavd_new, phiavd, phiav, int_step)
            phiavd = phiavd_new
        else:
            phiav = phiavc

        phiavx = phiav * DEG
        if abs(phiavx) >= philimx:
            phiavx = philimx * cadac_sign(phiavx)
        phiavout = phiavx * RAD

        ancomx = sqrt(acoma2 * acoma2 + acoma3 * acoma3) / grav
        if tanx:
            anxd_new = (ancomx - anx) / tanx
            anx = integrate(anxd_new, anxd, anx, int_step)
            anxd = anxd_new
        else:
            anx = ancomx
        if acft_option > 0:
            anlimx = pdynmc * clalpha * alplimx / wingloading
            if abs(anx) >= anlimx:
                anx = anlimx * cadac_sign(anx)

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


class Aim5AircraftGuidance:
    name = "guidance"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("acft_option", 0, "int", "data", "guidance"),
            Field("guid_gain", 0.0, "real", "data", "guidance"),
            Field("ACOML", (0.0, 0.0, 0.0), "vec", "out", "guidance"),
            Field("gturn", 0.0, "real", "data", "guidance"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        acft_option = store.get("acft_option")
        guid_gain = store.get("guid_gain")
        gturn = store.get("gturn")
        grav = store.get("grav")

        if acft_option == 0:
            acoml = np.array([0.0, 0.0, -grav])
        elif acft_option == 1:
            tvl = store.get("TVL")
            acomv = np.array([0.0, gturn * grav, -grav])
            acoml = tvl.T @ acomv
        elif acft_option == 2:
            combus = ctx.combus or ()
            packet = next((p for p in combus if p.type == "AIM5"), None)
            if packet is None:
                raise ValueError("no AIM5 packet on combus")
            stel = packet.vars["SBEL"]
            vtel = packet.vars["VBEL"]
            sbel = store.get("SBEL")
            vael = store.get("VBEL")
            satl = sbel - stel
            dab = float(np.linalg.norm(satl))
            gain = guid_gain * float(np.linalg.norm(skew(vael) @ vtel)) / dab
            uvtel = vtel / np.linalg.norm(vtel)
            uvael = vael / np.linalg.norm(vael)
            epsl = skew(uvael) @ uvtel
            acoml = skew(epsl) @ uvael * gain + np.array([0.0, 0.0, -grav])
        else:
            raise ValueError(f"unsupported acft_option={acft_option}")

        store.set("ACOML", acoml)

    def terminate(self, vehicle, ctx):
        pass


class Aim5Aircraft:
    type = "AIRCRAFT3"

    def __init__(self, name, events=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Flat3Environment(),
            Flat3Kinematics(),
            Aim5AircraftGuidance(),
            Aim5AircraftControl(),
            Aim5AircraftForces(),
            Aim5Flat3Newton(),
        ]

    def define(self):
        _define_aim5_vehicle(self)


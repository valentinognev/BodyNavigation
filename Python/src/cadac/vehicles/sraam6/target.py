from math import atan2, sqrt

import numpy as np

from cadac.constants import DEG, EPS, RAD
from cadac.eom.flat3 import (
    Flat3AircraftEnvironment,
    Flat3AircraftNewton,
    Flat3Kinematics,
)
from cadac.kernel.events import EventEngine
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore

_ZEROS3 = (0.0, 0.0, 0.0)


def _sign(variable):
    if variable < 0.0:
        return -1
    return 1


class Sraam6TargetGuidance:
    name = "guidance"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("msl_num", 0, "int", "data", "guidance"),
            Field("tgt_option", 0, "int", "data", "guidance"),
            Field("guid_gain", 0.0, "real", "data", "guidance"),
            Field("ACOML", _ZEROS3, "vec", "out", "guidance"),
            Field("gturn", 0.0, "real", "data", "guidance"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        msl_num = store.get("msl_num")
        tgt_option = store.get("tgt_option")
        guid_gain = store.get("guid_gain")
        gturn = store.get("gturn")
        grav = store.get("grav")
        tvl = np.asarray(store.get("TVL"), dtype=float)
        sael = np.asarray(store.get("SAEL"), dtype=float)
        vael = np.asarray(store.get("VAEL"), dtype=float)

        if tgt_option not in (0, 1, 2):
            raise ValueError(f"unknown tgt_option {tgt_option}")

        acoml = np.zeros(3)
        if tgt_option == 0:
            acoml = np.array([0.0, 0.0, -grav])
        if tgt_option == 1:
            acomv = np.array([0.0, gturn * grav, -grav])
            acoml = tvl.T @ acomv
        if tgt_option == 2:
            missiles = [
                packet
                for packet in (ctx.combus or ())
                if packet.type == "MISSILE6"
            ]
            if msl_num >= 1:
                idx = msl_num - 1
                if idx < len(missiles):
                    packet = missiles[idx]
                    mseek = int(packet.vars["mseek"])
                    if mseek % 10 == 4:
                        sbel = np.asarray(packet.vars["SBEL"], dtype=float)
                        vbel = np.asarray(packet.vars["VBEL"], dtype=float)
                        sabl = sael - sbel
                        dab = float(np.linalg.norm(sabl))
                        dum = float(np.linalg.norm(np.cross(vael, vbel)))
                        gain = guid_gain * dum / dab
                        uvbel = vbel * (1.0 / float(np.linalg.norm(vbel)))
                        uvael = vael * (1.0 / float(np.linalg.norm(vael)))
                        epsl = np.cross(uvael, uvbel)
                        acoml = np.cross(epsl, uvael) * gain
                        acoml = acoml + np.array([0.0, 0.0, -grav])

        store.set("ACOML", acoml)

    def terminate(self, vehicle, ctx):
        pass


class Sraam6TargetControl:
    name = "control"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("phiav", 0.0, "real", "state", "control"),
            Field("phiavd", 0.0, "real", "state", "control"),
            Field("tphi", 0.0, "real", "data", "control"),
            Field("philimx", 120.0, "real", "data", "control"),
            Field("phiavx", 0.0, "real", "out", "control", ("com",)),
            Field("phiavcx", 0.0, "real", "diag", "control"),
            Field("anx", 0.0, "real", "state", "control", ("com",)),
            Field("anxd", 0.0, "real", "state", "control"),
            Field("tanx", 0.0, "real", "data", "control"),
            Field("alplimx", 40.0, "real", "data", "control"),
            Field("ancomx", 0.0, "real", "diag", "control"),
            Field("clalpha", 0.0523, "real", "data", "control"),
            Field("wingloading", 3247.0, "real", "data", "control"),
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
        tgt_option = store.get("tgt_option")
        acoml = np.asarray(store.get("ACOML"), dtype=float)
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
            phiavx = philimx * _sign(phiavx)

        phiavout = phiavx * RAD

        ancomx = sqrt(acoma2 * acoma2 + acoma3 * acoma3) / grav

        if tanx:
            anxd_new = (ancomx - anx) / tanx
            anx = integrate(anxd_new, anxd, anx, int_step)
            anxd = anxd_new
        else:
            anx = ancomx
        if tgt_option > 0:
            anlimx = pdynmc * clalpha * alplimx / (wingloading * grav)
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


class Sraam6TargetForces:
    name = "forces"

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("FSPA", _ZEROS3, "vec", "out", "forces"),
            Field("acc_longx", 0.0, "real", "data", "forces"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        acc_longx = store.get("acc_longx")
        grav = store.get("grav")
        anx = store.get("anx")
        acoma1 = acc_longx * grav
        acoma2 = 0.0
        acoma3 = -anx * grav
        store.set("FSPA", np.array([acoma1, acoma2, acoma3]))

    def terminate(self, vehicle, ctx):
        pass


class Sraam6Target:
    type = "TARGET3"

    def __init__(self, name, events=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Flat3AircraftEnvironment(),
            Flat3Kinematics(),
            Flat3AircraftNewton(),
            Sraam6TargetGuidance(),
            Sraam6TargetControl(),
            Sraam6TargetForces(),
        ]

    def define(self):
        store = self.store
        orig_define = store.define

        def define_skip_if_exists(field):
            if field.name not in store.names():
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

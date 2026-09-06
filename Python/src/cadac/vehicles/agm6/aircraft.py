from math import atan2, cos, fabs, sin, sqrt

import numpy as np

from cadac.constants import DEG, EPS, RAD
from cadac.eom.flat3 import Flat3Kinematics
from cadac.kernel.events import EventEngine
from cadac.kernel.integrate import integrate
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import polar_from_cart
from cadac.vehicles.agm6.flat3io import Agm6Flat3Environment, Agm6Flat3Newton


def _sign(variable):
    if variable < 0:
        return -1
    return 1


def _skew(vec):
    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def _cart_from_pol(magnitude, azimuth, elevation):
    return np.array(
        [
            magnitude * (cos(elevation) * cos(azimuth)),
            magnitude * (cos(elevation) * sin(azimuth)),
            magnitude * (sin(elevation) * (-1.0)),
        ],
        dtype=float,
    )


def _vec3(vars_, primary, fallback):
    if primary in vars_:
        return np.array(vars_[primary], dtype=float, copy=True)
    if fallback in vars_:
        return np.array(vars_[fallback], dtype=float, copy=True)
    return np.zeros(3)


class Agm6AircraftGuidance:
    name = "guidance"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        for field in (
            Field("acft_option", 0, "int", "data", "guidance"),
            Field("guid_gain", 0.0, "real", "data", "guidance"),
            Field("ACOML", zeros3, "vec", "out", "guidance"),
            Field("gturn", 0.0, "real", "data", "guidance"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        acft_option = store.get("acft_option")
        grav = store.get("grav")
        if acft_option == 0:
            acoml = np.array([0.0, 0.0, -grav], dtype=float)
        elif acft_option == 1:
            tvl = np.asarray(store.get("TVL"), dtype=float)
            gturn = store.get("gturn")
            acomv = np.array([0.0, gturn * grav, -grav], dtype=float)
            acoml = tvl.T @ acomv
        elif acft_option == 2:
            guid_gain = store.get("guid_gain")
            sael = np.asarray(store.get("SAEL"), dtype=float)
            vael = np.asarray(store.get("VAEL"), dtype=float)
            stel = np.zeros(3)
            vtel = np.zeros(3)
            for packet in ctx.combus or ():
                if packet is not None and packet.type == "TARGET3":
                    vars_ = packet.vars
                    stel = np.array(vars_["SAEL"], dtype=float, copy=True)
                    vtel = np.array(vars_["VAEL"], dtype=float, copy=True)
                    break
            satl = sael - stel
            dab = float(np.linalg.norm(satl))
            dum = float(np.linalg.norm(_skew(vael) @ vtel))
            gain = guid_gain * dum / dab
            uvtel = vtel * (1.0 / float(np.linalg.norm(vtel)))
            uvael = vael * (1.0 / float(np.linalg.norm(vael)))
            epsl = _skew(uvael) @ uvtel
            acoml = _skew(epsl) @ uvael * gain
            acoml = acoml + np.array([0.0, 0.0, -grav], dtype=float)
        else:
            raise ValueError(f"unknown acft_option {acft_option}")
        store.set("ACOML", acoml)

    def terminate(self, vehicle, ctx):
        pass


class Agm6AircraftControl:
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
        int_step = ctx.int_step

        acomv = tvl @ acoml
        acoma2 = acomv[1]
        acoma3 = acomv[2]
        if fabs(acoma2) < EPS and fabs(acoma3) < EPS:
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
        if fabs(phiavx) >= philimx:
            phiavx = philimx * _sign(phiavx)
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
            if fabs(anx) >= anlimx:
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


class Agm6AircraftForces:
    name = "forces"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        store.define(Field("FSPA", zeros3, "vec", "out", "forces"))
        if "FSPV" not in store.names():
            store.define(Field("FSPV", zeros3, "vec", "out", "forces"))
        store.define(Field("acc_longx", 0.0, "real", "data", "forces"))

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        acc_longx = store.get("acc_longx")
        grav = store.get("grav")
        anx = store.get("anx")
        fspa = np.array([acc_longx * grav, 0.0, -anx * grav], dtype=float)
        store.set("FSPA", fspa)
        store.set("FSPV", fspa)

    def terminate(self, vehicle, ctx):
        pass


class Agm6AircraftSensor:
    name = "sensor"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        com = ("com",)
        for field in (
            Field("init_flag", 1, "int", "data", "sensor"),
            Field("track_epoch", 0.0, "real", "save", "sensor"),
            Field("track_step", 0.0, "real", "data", "sensor"),
            Field("target_num", 0, "int", "save", "sensor"),
            Field("STCEL1", zeros3, "vec", "out", "sensor", com),
            Field("VTCEL1", zeros3, "vec", "out", "sensor", com),
            Field("STCEL2", zeros3, "vec", "out", "sensor", com),
            Field("VTCEL2", zeros3, "vec", "out", "sensor", com),
            Field("STCEL3", zeros3, "vec", "out", "sensor", com),
            Field("VTCEL3", zeros3, "vec", "out", "sensor", com),
            Field("STCEL4", zeros3, "vec", "out", "sensor"),
            Field("VTCEL4", zeros3, "vec", "out", "sensor"),
            Field("STCEL5", zeros3, "vec", "out", "sensor"),
            Field("VTCEL5", zeros3, "vec", "out", "sensor"),
            Field("dat_sigma", 0.0, "real", "data", "sensor"),
            Field("azat_sigma", 0.0, "real", "data", "sensor"),
            Field("elat_sigma", 0.0, "real", "data", "sensor"),
            Field("vel_sigma", 0.0, "real", "data", "sensor"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        init_flag = store.get("init_flag")
        target_num = store.get("target_num")
        track_step = store.get("track_step")
        dat_sigma = store.get("dat_sigma")
        azat_sigma = store.get("azat_sigma")
        elat_sigma = store.get("elat_sigma")
        vel_sigma = store.get("vel_sigma")
        track_epoch = store.get("track_epoch")
        stcel = [
            np.array(store.get(f"STCEL{i}"), dtype=float, copy=True)
            for i in range(1, 6)
        ]
        vtcel = [
            np.array(store.get(f"VTCEL{i}"), dtype=float, copy=True)
            for i in range(1, 6)
        ]
        sael = np.asarray(store.get("SAEL"), dtype=float)

        if init_flag:
            init_flag = 0
            track_epoch = ctx.sim_time

        if ctx.sim_time >= track_epoch:
            track_epoch = ctx.sim_time + track_step
            target_num = 1
            for packet in ctx.combus or ():
                if packet is None or packet.type != "TARGET3":
                    continue
                stel = _vec3(packet.vars, "SAEL", "SBEL")
                vtel = _vec3(packet.vars, "VAEL", "VBEL")
                satl = sael - stel
                polar = polar_from_cart(satl)
                satcl = _cart_from_pol(
                    float(polar[0]) + dat_sigma,
                    float(polar[1]) + azat_sigma,
                    float(polar[2]) + elat_sigma,
                )
                idx = target_num - 1
                stcel[idx] = sael - satcl
                vtcel[idx] = np.array(
                    [
                        vtel[0] + vel_sigma,
                        vtel[1] + vel_sigma,
                        vtel[2] + vel_sigma,
                    ],
                    dtype=float,
                )
                target_num += 1
                if target_num > 5:
                    break

        store.set("init_flag", init_flag)
        store.set("track_epoch", track_epoch)
        store.set("target_num", target_num)
        for i in range(1, 6):
            store.set(f"STCEL{i}", stcel[i - 1])
            store.set(f"VTCEL{i}", vtcel[i - 1])

    def terminate(self, vehicle, ctx):
        pass


class Agm6Aircraft:
    type = "AIRCRAFT3"

    def __init__(self, name, events=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Agm6Flat3Environment(),
            Flat3Kinematics(),
            Agm6AircraftGuidance(),
            Agm6AircraftControl(),
            Agm6AircraftForces(),
            Agm6Flat3Newton(),
            Agm6AircraftSensor(),
        ]

    def define(self):
        for module in self.modules:
            module.define(self)
        self.com_names = [
            name
            for name in self.store.names()
            if "com" in self.store.field(name).outputs
        ]

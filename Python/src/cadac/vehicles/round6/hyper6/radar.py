import math

import numpy as np

from cadac.constants import RAD, REARTH, WEII3
from cadac.kernel.events import EventEngine
from cadac.kernel.state import Field, StateStore
from cadac.math.earth import cadtei, cadtge
from cadac.math.frames import cadac_matmul, cart_from_pol, polar_from_cart
from cadac.stoch import gauss

# C++ global_constants.hpp: size of the ground0[] module-variable array.
NGROUND0 = 20
_TRACKS = 5


def _ground_kinematics(lonx, latx, alt, sim_time):
    """CADAC ``Ground0::kinematics``: Earth-fixed point in inertial coordinates."""
    weii = np.zeros((3, 3))
    weii[0, 1] = -WEII3
    weii[1, 0] = WEII3
    sbig = np.array([0.0, 0.0, -(alt + REARTH)])
    tge = cadtge(lonx * RAD, latx * RAD)
    teg = tge.T
    sbie = cadac_matmul(teg, sbig)
    tei = cadtei(sim_time)
    sbii = cadac_matmul(tei.T, sbie)
    dbi = float(math.sqrt(sbii[0] * sbii[0] + sbii[1] * sbii[1] + sbii[2] * sbii[2]))
    vbii = cadac_matmul(weii, sbii)
    tig = cadac_matmul(tei.T, teg)
    return sbii, vbii, dbi, tig, tge


class Hyper6GroundKinematics:
    """Ground0 kinematics owned by the radar. Ground0 is not a vehicle."""

    name = "kinematics"

    def define(self, vehicle):
        store = vehicle.store
        store.define(Field("time", 0.0, "real", "diag", "kinematics"))
        store.define(Field("lonx", 0.0, "real", "data", "kinematics"))
        store.define(Field("latx", 0.0, "real", "data", "kinematics"))
        store.define(Field("alt", 0.0, "real", "data", "kinematics"))
        store.define(Field("sbii", (0.0, 0.0, 0.0), "vec", "state", "kinematics", ("com",)))
        store.define(Field("vbii", (0.0, 0.0, 0.0), "vec", "state", "kinematics", ("com",)))
        store.define(Field("dbi", 0.0, "real", "diag", "kinematics", ("com",)))
        store.define(
            Field(
                "tig",
                ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                "mat",
                "init",
                "kinematics",
            )
        )
        store.define(
            Field(
                "tge",
                ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                "mat",
                "out",
                "kinematics",
            )
        )
        store.define(Field("NGROUND0", NGROUND0, "int", "diag", "kinematics"))

    def initialize(self, vehicle, ctx):
        self._load(vehicle.store, ctx.sim_time)

    def execute(self, vehicle, ctx):
        self._load(vehicle.store, ctx.sim_time)

    def _load(self, store, sim_time):
        sbii, vbii, dbi, tig, tge = _ground_kinematics(
            store.get("lonx"),
            store.get("latx"),
            store.get("alt"),
            sim_time,
        )
        store.set("time", sim_time)
        store.set("sbii", sbii)
        store.set("vbii", vbii)
        store.set("dbi", dbi)
        store.set("tig", tig)
        store.set("tge", tge)


class Hyper6RadarSeeker:
    """CADAC ``Radar::seeker``: up to five satellite track files on combus."""

    name = "seeker"

    def define(self, vehicle):
        store = vehicle.store
        zeros = (0.0, 0.0, 0.0)
        store.define(Field("radar_on", 0, "int", "data", "seeker"))
        store.define(Field("init_flag", 1, "int", "data", "seeker"))
        store.define(Field("track_epoch", 0.0, "real", "save", "seeker"))
        store.define(Field("track_step", 0.0, "real", "data", "seeker"))
        store.define(Field("target_num", 0, "int", "save", "seeker"))
        for index in range(1, _TRACKS + 1):
            store.define(
                Field(f"stcii{index}", zeros, "vec", "save", "seeker", ("com",))
            )
            store.define(
                Field(f"vtcii{index}", zeros, "vec", "save", "seeker", ("com",))
            )
        store.define(Field("dat_sigma", 0.0, "real", "data", "seeker"))
        store.define(Field("azat_sigma", 0.0, "real", "data", "seeker"))
        store.define(Field("elat_sigma", 0.0, "real", "data", "seeker"))
        store.define(Field("vel_sigma", 0.0, "real", "data", "seeker"))

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        if not store.get("radar_on"):
            return
        init_flag = store.get("init_flag")
        track_epoch = store.get("track_epoch")
        if init_flag:
            init_flag = 0
            track_epoch = ctx.sim_time
        if ctx.sim_time >= track_epoch:
            track_epoch = ctx.sim_time + store.get("track_step")
            target_num = _measure_tracks(store, ctx.combus)
        else:
            target_num = store.get("target_num")
        store.set("init_flag", init_flag)
        store.set("track_epoch", track_epoch)
        store.set("target_num", target_num)


def _measure_tracks(store, combus):
    """First five combus satellites, ids ``t1``..``t5`` as in C++."""
    target_num = 1
    dat_sigma = store.get("dat_sigma")
    azat_sigma = store.get("azat_sigma")
    elat_sigma = store.get("elat_sigma")
    vel_sigma = store.get("vel_sigma")
    sbii = np.array(store.get("sbii"), dtype=float)
    packets = list(combus) if combus is not None else []
    for packet in packets:
        if target_num > _TRACKS:
            break
        ident = getattr(packet, "id", None) or ""
        if not ident and getattr(packet, "type", "") == "SAT3":
            ident = f"t{target_num}"
        if ident != f"t{target_num}":
            continue
        vars_ = packet.vars
        stii = np.array(vars_["sbii"], dtype=float)
        vtii = np.array(vars_["vbii"], dtype=float)
        polar = polar_from_cart(sbii - stii)
        dat_meas = float(polar[0]) + gauss(0.0, dat_sigma)
        azat_meas = float(polar[1]) + gauss(0.0, azat_sigma)
        elat_meas = float(polar[2]) + gauss(0.0, elat_sigma)
        sbtci = cart_from_pol(dat_meas, azat_meas, elat_meas)
        vtcii = np.array(
            [
                vtii[0] + gauss(0.0, vel_sigma),
                vtii[1] + gauss(0.0, vel_sigma),
                vtii[2] + gauss(0.0, vel_sigma),
            ],
            dtype=float,
        )
        store.set(f"stcii{target_num}", sbii - sbtci)
        store.set(f"vtcii{target_num}", vtcii)
        target_num += 1
    return target_num


class Hyper6Radar:
    type = "RADAR0"

    def __init__(self, name, events=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Hyper6GroundKinematics(),
            Hyper6RadarSeeker(),
        ]

    def define(self):
        for module in self.modules:
            module.define(self)
        self.com_names = [
            name
            for name in self.store.names()
            if "com" in self.store.field(name).outputs
        ]

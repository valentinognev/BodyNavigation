from math import cos, sin

import numpy as np

from cadac.eom.flat0 import Flat0Kinematics, Flat0Newton
from cadac.kernel.events import EventEngine
from cadac.kernel.state import Field, StateStore
from cadac.math.frames import polar_from_cart

SMALL = 1e-7


def _cart_from_pol(magnitude, azimuth, elevation):
    return np.array(
        [
            magnitude * (cos(elevation) * cos(azimuth)),
            magnitude * (cos(elevation) * sin(azimuth)),
            magnitude * (sin(elevation) * (-1.0)),
        ],
        dtype=float,
    )


def _packets_of_type(combus, type_name):
    if not combus:
        return []
    return [packet for packet in combus if packet.type == type_name]


class Sam6RadarSensor:
    name = "sensor"

    def __init__(self, sam_deck=None, srmb_deck=None):
        self.missile_traj = sam_deck
        self.rocket_traj = srmb_deck

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        com = ("com",)
        for field in (
            Field("mtrack", 0, "int", "data", "sensor"),
            Field("alt_engage", 0.0, "real", "data", "sensor"),
            Field("init_flag", 1, "int", "init", "sensor"),
            Field("track_epoch", 0.0, "real", "save", "sensor"),
            Field("track_step", 0.0, "real", "data", "sensor"),
            Field("rocket_num", 0, "int", "save", "sensor"),
            Field("launch_delay", 9999.0, "real", "save", "sensor"),
            Field("SIEL", zeros3, "vec", "save", "sensor"),
            Field("ip_alt_bias", 0.0, "real", "data", "sensor"),
            Field("lnch_dly_bias1", 0.0, "real", "data", "sensor"),
            Field("lnch_dly_bias2", 0.0, "real", "data", "sensor"),
            Field("lnch_dly_bias3", 0.0, "real", "data", "sensor"),
            Field("dat_sigma", 0.0, "real", "data", "sensor"),
            Field("azat_sigma", 0.0, "real", "data", "sensor"),
            Field("elat_sigma", 0.0, "real", "data", "sensor"),
            Field("vel_sigma", 0.0, "real", "data", "sensor"),
            Field("apo_flag", 0, "int", "save", "sensor"),
            Field("apo_epoch", 0.0, "real", "save", "sensor"),
            Field("lnch_delay_m1", 0.0, "real", "out", "sensor", com),
            Field("lnch_delay_m2", 0.0, "real", "out", "sensor", com),
            Field("lnch_delay_m3", 0.0, "real", "out", "sensor", com),
            Field("SIEL1", zeros3, "vec", "out", "sensor", com),
            Field("SIEL2", zeros3, "vec", "out", "sensor", com),
            Field("SIEL3", zeros3, "vec", "out", "sensor", com),
            Field("aircraft_num", 0, "int", "save", "sensor"),
            Field("lethal_rng", 0.0, "real", "data", "sensor"),
            Field("lethal_flag1", 0, "int", "save", "sensor"),
            Field("lethal_flag2", 0, "int", "save", "sensor"),
            Field("lethal_flag3", 0, "int", "save", "sensor"),
            Field("launch_delay1", 9999.0, "real", "save", "sensor"),
            Field("launch_delay2", 9999.0, "real", "save", "sensor"),
            Field("launch_delay3", 9999.0, "real", "save", "sensor"),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mtrack = store.get("mtrack")
        if mtrack not in (0, 1, 2):
            raise ValueError(f"mtrack={mtrack!r} not supported")
        if mtrack == 0:
            return

        alt_engage = store.get("alt_engage")
        init_flag = store.get("init_flag")
        rocket_num = store.get("rocket_num")
        track_step = store.get("track_step")
        ip_alt_bias = store.get("ip_alt_bias")
        biases = (
            store.get("lnch_dly_bias1"),
            store.get("lnch_dly_bias2"),
            store.get("lnch_dly_bias3"),
        )
        aircraft_num = store.get("aircraft_num")
        lethal_rng = store.get("lethal_rng")
        track_epoch = store.get("track_epoch")
        launch_delay = store.get("launch_delay")
        siel = np.asarray(store.get("SIEL"), dtype=float).copy()
        apo_flag = store.get("apo_flag")
        apo_epoch = store.get("apo_epoch")
        lnch_m = [
            store.get("lnch_delay_m1"),
            store.get("lnch_delay_m2"),
            store.get("lnch_delay_m3"),
        ]
        siel_m = [
            np.asarray(store.get("SIEL1"), dtype=float).copy(),
            np.asarray(store.get("SIEL2"), dtype=float).copy(),
            np.asarray(store.get("SIEL3"), dtype=float).copy(),
        ]
        lethal_flags = [
            store.get("lethal_flag1"),
            store.get("lethal_flag2"),
            store.get("lethal_flag3"),
        ]
        launch_delays = [
            store.get("launch_delay1"),
            store.get("launch_delay2"),
            store.get("launch_delay3"),
        ]
        srel = np.asarray(store.get("SREL"), dtype=float)

        sim_time = ctx.sim_time
        if init_flag:
            init_flag = 0
            track_epoch = sim_time

        if mtrack == 1:
            (
                rocket_num,
                track_epoch,
                launch_delay,
                siel,
                apo_flag,
                apo_epoch,
                lnch_m,
                siel_m,
            ) = self._track_rockets(
                ctx,
                sim_time,
                track_epoch,
                track_step,
                srel,
                alt_engage,
                ip_alt_bias,
                biases,
                rocket_num,
                launch_delay,
                siel,
                apo_flag,
                apo_epoch,
                lnch_m,
                siel_m,
            )
        else:
            (
                aircraft_num,
                track_epoch,
                lethal_flags,
                launch_delays,
                lnch_m,
                siel_m,
            ) = self._track_aircraft(
                ctx,
                sim_time,
                track_epoch,
                track_step,
                srel,
                lethal_rng,
                biases,
                aircraft_num,
                lethal_flags,
                launch_delays,
                lnch_m,
                siel_m,
            )

        store.set("lnch_delay_m1", lnch_m[0])
        store.set("lnch_delay_m2", lnch_m[1])
        store.set("lnch_delay_m3", lnch_m[2])
        store.set("SIEL1", siel_m[0])
        store.set("SIEL2", siel_m[1])
        store.set("SIEL3", siel_m[2])
        store.set("init_flag", init_flag)
        store.set("track_epoch", track_epoch)
        store.set("rocket_num", rocket_num)
        store.set("launch_delay", launch_delay)
        store.set("SIEL", siel)
        store.set("apo_flag", apo_flag)
        store.set("apo_epoch", apo_epoch)
        store.set("aircraft_num", aircraft_num)
        store.set("lethal_flag1", lethal_flags[0])
        store.set("lethal_flag2", lethal_flags[1])
        store.set("lethal_flag3", lethal_flags[2])
        store.set("launch_delay1", launch_delays[0])
        store.set("launch_delay2", launch_delays[1])
        store.set("launch_delay3", launch_delays[2])

    def terminate(self, vehicle, ctx):
        pass

    def _measure(self, stel, srel):
        strl = np.asarray(stel, dtype=float) - srel
        polar = polar_from_cart(strl)
        dat = float(polar[0])
        azat = float(polar[1])
        elat = float(polar[2])
        strcl = _cart_from_pol(dat, azat, elat)
        stcel = strcl - srel
        dtrc = float(np.linalg.norm(strcl))
        return strcl, stcel, dtrc

    def _track_aircraft(
        self,
        ctx,
        sim_time,
        track_epoch,
        track_step,
        srel,
        lethal_rng,
        biases,
        aircraft_num,
        lethal_flags,
        launch_delays,
        lnch_m,
        siel_m,
    ):
        if sim_time < track_epoch:
            return (
                aircraft_num,
                track_epoch,
                lethal_flags,
                launch_delays,
                lnch_m,
                siel_m,
            )
        track_epoch = sim_time + track_step
        aircraft = _packets_of_type(ctx.combus, "AIRCRAFT3")[:3]
        aircraft_num = 1
        for packet in aircraft:
            stel = packet.vars["SAEL"]
            _strcl, stcel, dtrc = self._measure(stel, srel)
            k = aircraft_num - 1
            if (dtrc < lethal_rng) and (not lethal_flags[k]):
                lethal_flags[k] = 1
                launch_delays[k] = sim_time
            if lethal_flags[k]:
                lnch_m[k] = launch_delays[k] + biases[k]
            siel_m[k] = stcel
            aircraft_num += 1
            if aircraft_num > 3:
                break
        return (
            aircraft_num,
            track_epoch,
            lethal_flags,
            launch_delays,
            lnch_m,
            siel_m,
        )

    def _track_rockets(
        self,
        ctx,
        sim_time,
        track_epoch,
        track_step,
        srel,
        alt_engage,
        ip_alt_bias,
        biases,
        rocket_num,
        launch_delay,
        siel,
        apo_flag,
        apo_epoch,
        lnch_m,
        siel_m,
    ):
        if sim_time < track_epoch:
            return (
                rocket_num,
                track_epoch,
                launch_delay,
                siel,
                apo_flag,
                apo_epoch,
                lnch_m,
                siel_m,
            )
        if self.rocket_traj is None or self.missile_traj is None:
            raise ValueError("mtrack=1 requires sam_deck and srmb_deck")
        track_epoch = sim_time + track_step
        rockets = _packets_of_type(ctx.combus, "ROCKET5")[:3]
        missiles = _packets_of_type(ctx.combus, "MISSILE6")
        rocket_num = 1
        rocket_traj = self.rocket_traj
        missile_traj = self.missile_traj
        for packet in rockets:
            stel = packet.vars["SAEL"]
            vtel = np.asarray(packet.vars["VAEL"], dtype=float)
            _strcl, stcel, _dtrc = self._measure(stel, srel)
            vtcel = np.asarray(vtel, dtype=float).copy()
            vtcel3 = float(vtcel[2])
            alt_diff_rock = 0.0
            alt_diff_misl = 0.0
            if vtcel3 > 0.0 and not apo_flag:
                apo_flag = 1
                apo_epoch = sim_time
                ip_apo_time_rocket = rocket_traj.look_up(
                    "apotime_vs_descent_altitude", alt_engage
                )
                ip_time_missile = missile_traj.look_up(
                    "time_vs_ascent_altitude", alt_engage
                )
                launch_delay = apo_epoch + ip_apo_time_rocket - ip_time_missile
                siel1 = rocket_traj.look_up(
                    "x_vs_launch_time", ip_apo_time_rocket + apo_epoch
                )
                siel2 = rocket_traj.look_up(
                    "y_vs_launch_time", ip_apo_time_rocket + apo_epoch
                )
                siel = np.array([siel1, siel2, -alt_engage], dtype=float)
            if sim_time > launch_delay:
                alt_rock_actual = -float(stcel[2])
                apo_time_rocket = rocket_traj.look_up(
                    "apotime_vs_descent_altitude", alt_rock_actual
                )
                alt_rock_predicted = -rocket_traj.look_up(
                    "z_vs_launch_time", apo_time_rocket + apo_epoch
                )
                alt_diff_rock = alt_rock_predicted - alt_rock_actual
                sbel = np.zeros(3)
                k = rocket_num - 1
                if k < len(missiles) and "SBEL" in missiles[k].vars:
                    sbel = np.asarray(missiles[k].vars["SBEL"], dtype=float)
                _sbrcl, sbcel, _ = self._measure(sbel, srel)
                alt_misl_actual = -float(sbcel[2])
                time_missile = missile_traj.look_up(
                    "time_vs_ascent_altitude", alt_misl_actual
                )
                alt_misl_predicted = missile_traj.look_up(
                    "alt_vs_launch_time", time_missile
                )
                alt_diff_misl = alt_misl_predicted - alt_misl_actual
            k = rocket_num - 1
            lnch_m[k] = launch_delay + biases[k]
            delta_ip = -alt_diff_rock - alt_diff_misl
            alt_ip = -float(siel[2])
            alt_ip += delta_ip
            ip_apo_time_rocket = rocket_traj.look_up(
                "apotime_vs_descent_altitude", alt_ip
            )
            siel1 = rocket_traj.look_up(
                "x_vs_launch_time", ip_apo_time_rocket + apo_epoch
            )
            siel2 = rocket_traj.look_up(
                "y_vs_launch_time", ip_apo_time_rocket + apo_epoch
            )
            siel3 = -alt_ip - ip_alt_bias
            siel_m[k] = np.array([siel1, siel2, siel3], dtype=float)
            rocket_num += 1
            if rocket_num > 3:
                break
        return (
            rocket_num,
            track_epoch,
            launch_delay,
            siel,
            apo_flag,
            apo_epoch,
            lnch_m,
            siel_m,
        )


class Sam6Radar:
    type = "RADAR0"

    def __init__(self, name, events=None, sam_deck=None, srmb_deck=None):
        self.name = name
        self.health = 1
        self.store = StateStore()
        self.event_time = 0.0
        self.events = EventEngine(events or [])
        self.com_names = []
        self.modules = [
            Sam6RadarSensor(sam_deck, srmb_deck),
            Flat0Kinematics(),
            Flat0Newton(),
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

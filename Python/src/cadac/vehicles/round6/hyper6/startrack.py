"""HYPER6 star tracker — port of Hyper::startrack (startrack.cpp)."""

import math

import numpy as np

from cadac.constants import RAD
from cadac.kernel.state import Field
from cadac.math.frames import cadac_inverse, cadac_matmul, cart_from_pol, polar_from_cart, skew

# 25 bright star catalog, J2000 unit vectors (Hyper::startrack_init).
STAR_CATALOG = np.array(
    [
        [-0.179457, 0.947482, -0.264715],
        [-0.062053, 0.621699, -0.780794],
        [-0.395709, -0.321011, -0.860446],
        [-0.787739, -0.518432, 0.332710],
        [0.119339, -0.770884, 0.625697],
        [0.205167, 0.969392, -0.134851],
        [0.141589, 0.680716, 0.718733],
        [-0.407741, 0.908325, 0.093239],
        [0.504096, 0.224174, -0.834046],
        [-0.434428, -0.251576, -0.864859],
        [0.449879, -0.880088, 0.151836],
        [0.355451, 0.890944, 0.282620],
        [-0.479412, -0.049965, -0.876167],
        [0.032446, 0.991140, 0.128796],
        [-0.357911, -0.827083, -0.433397],
        [-0.923975, -0.348220, -0.158158],
        [-0.380630, 0.795326, 0.471781],
        [0.846646, -0.247176, -0.471268],
        [0.453017, -0.541322, 0.708340],
        [-0.511331, -0.101246, -0.853399],
        [-0.858250, 0.467448, 0.211893],
        [-0.217999, 0.863108, -0.455545],
        [0.161912, 0.980685, 0.109734],
        [-0.103643, -0.792588, -0.600885],
        [0.140796, 0.866902, 0.478181],
    ],
    dtype=float,
)
STAR_NAMES = [
    "Sirius",
    "Canopus",
    "Rigil Kent",
    "Arcturus",
    "Vega",
    "Rigel",
    "Capella",
    "Procyon",
    "Achernar",
    "Hadar",
    "Altair",
    "Aslebaran",
    "Acrux",
    "Betelgeuse",
    "Antares",
    "Spica",
    "Pollux",
    "Fomalhout",
    "Deneb",
    "Mimosa",
    "Regulus",
    "Adhara",
    "Bellatrix",
    "Shaula",
    "El Nath",
]


def star_init():
    return np.array(STAR_CATALOG, dtype=float, copy=True), list(STAR_NAMES)


def _univec3(vec):
    d = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    if d == 0.0:
        return np.zeros(3)
    return np.array([vec[0] / d, vec[1] / d, vec[2] / d], dtype=float)


def star_triad(star_data, star_el_min, sbii):
    star_usii = np.asarray(star_data, dtype=float)
    sbii = np.asarray(sbii, dtype=float)
    ubii = _univec3(sbii)
    vis_star_slot = np.zeros(25, dtype=int)
    visible_count = 0
    for i in range(25):
        usii = star_usii[i]
        elstar = math.asin(usii[0] * ubii[0] + usii[1] * ubii[1] + usii[2] * ubii[2])
        vis_star_slot[i] = 0
        if elstar > star_el_min * RAD:
            visible_count += 1
            vis_star_slot[i] = i + 1

    usii_triad = np.zeros((3, 4))
    star_volume = 0.0
    if visible_count < 3:
        return usii_triad, star_volume

    usii_vis = np.zeros((visible_count, 3))
    vis_star_slot_vis = np.zeros(visible_count, dtype=int)
    k = 0
    kk = 0
    for i in range(25):
        if vis_star_slot[i] > 0:
            usii_vis[k] = star_usii[i]
            k += 1
            vis_star_slot_vis[kk] = vis_star_slot[i]
            kk += 1

    triad = [0, 0, 0]
    nm2 = visible_count - 2
    nm1 = visible_count - 1
    for i1 in range(nm2):
        for i2 in range(i1 + 1, nm1):
            for i3 in range(i2 + 1, visible_count):
                usii1 = usii_vis[i1]
                usii2 = usii_vis[i2]
                usii3 = usii_vis[i3]
                crossed = cadac_matmul(skew(usii2), usii3)
                volume_local = abs(
                    float(
                        usii1[0] * crossed[0]
                        + usii1[1] * crossed[1]
                        + usii1[2] * crossed[2]
                    )
                )
                if volume_local > star_volume:
                    star_volume = volume_local
                    triad[0] = i1
                    triad[1] = i2
                    triad[2] = i3

    for m in range(3):
        usii_triad[m, :3] = usii_vis[triad[m]]
        usii_triad[m, 3] = vis_star_slot_vis[triad[m]]
    return usii_triad, star_volume


class Hyper6Startrack:
    name = "startrack"

    def __init__(self):
        self.star_data, self.star_names = star_init()

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        plot = ("plot",)
        for field in (
            Field("mstar", 0, "int", "data", "startrack"),
            Field("star_el_min", 0.0, "real", "data", "startrack"),
            Field("star_volume", 0.0, "real", "diag", "startrack"),
            Field("startrack_alt", 0.0, "real", "data", "startrack"),
            Field("star_acqtime", 0.0, "real", "data", "startrack"),
            Field("star_step", 0.0, "real", "data", "startrack"),
            Field("starfix_epoch", 0.0, "real", "save", "startrack"),
            Field("star_acq", 0, "int", "save", "startrack"),
            Field("star_slotsum", 0.0, "real", "save", "startrack"),
            Field("az1_bias", 0.0, "real", "data", "startrack"),
            Field("az2_bias", 0.0, "real", "data", "startrack"),
            Field("az3_bias", 0.0, "real", "data", "startrack"),
            Field("az1_noise", 0.0, "real", "data", "startrack"),
            Field("az2_noise", 0.0, "real", "data", "startrack"),
            Field("az3_noise", 0.0, "real", "data", "startrack"),
            Field("el1_bias", 0.0, "real", "data", "startrack"),
            Field("el2_bias", 0.0, "real", "data", "startrack"),
            Field("el3_bias", 0.0, "real", "data", "startrack"),
            Field("el1_noise", 0.0, "real", "data", "startrack"),
            Field("el2_noise", 0.0, "real", "data", "startrack"),
            Field("el3_noise", 0.0, "real", "data", "startrack"),
            Field("URIC", zeros3, "vec", "out", "startrack", plot),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mstar = store.get("mstar")
        if mstar not in (0, 1, 2, 3):
            raise ValueError(f"unknown mstar {mstar}")
        if mstar == 0:
            return

        time = store.get("time")
        alt = store.get("alt")
        star_el_min = store.get("star_el_min")
        startrack_alt = store.get("startrack_alt")
        star_acqtime = store.get("star_acqtime")
        star_step = store.get("star_step")
        az_bias = np.array(
            [store.get("az1_bias"), store.get("az2_bias"), store.get("az3_bias")],
            dtype=float,
        )
        az_noise = np.array(
            [store.get("az1_noise"), store.get("az2_noise"), store.get("az3_noise")],
            dtype=float,
        )
        el_bias = np.array(
            [store.get("el1_bias"), store.get("el2_bias"), store.get("el3_bias")],
            dtype=float,
        )
        el_noise = np.array(
            [store.get("el1_noise"), store.get("el2_noise"), store.get("el3_noise")],
            dtype=float,
        )
        starfix_epoch = store.get("starfix_epoch")
        star_acq = store.get("star_acq")
        star_slotsum = store.get("star_slotsum")
        uric = np.asarray(store.get("URIC"), dtype=float).copy()
        star_volume = store.get("star_volume")

        if mstar == 1:
            self.star_data, self.star_names = star_init()
            star_acq = 1
            starfix_epoch = time
            if alt > startrack_alt:
                mstar = 2
        if mstar == 2:
            if star_acq:
                dtime_star = star_acqtime
            else:
                dtime_star = star_step
            time_star = time - starfix_epoch
            if time_star >= dtime_star and alt > startrack_alt:
                mstar = 3
        if mstar == 3:
            star_acq = 0
            starfix_epoch = time
            sbii = np.asarray(store.get("SBII"), dtype=float)
            tbi = np.asarray(store.get("TBI"), dtype=float)
            tbic = np.asarray(store.get("TBIC"), dtype=float)
            usii_triad, star_volume = star_triad(
                self.star_data, star_el_min, sbii
            )
            triad_meas = np.zeros((3, 3))
            triad_true = np.zeros((3, 3))
            slot = np.zeros(3)
            slotm = 0.0
            for i in range(3):
                usii = usii_triad[i, :3]
                usib = cadac_matmul(tbi, usii)
                polar = polar_from_cart(usib)
                az = polar[1]
                el = polar[2]
                az_meas = az + az_bias[i] + az_noise[i]
                el_meas = el + el_bias[i] + el_noise[i]
                usibm = cart_from_pol(1.0, az_meas, el_meas)
                usiim = cadac_matmul(tbic.T.copy(), usibm)
                triad_meas[:, i] = usiim
                triad_true[:, i] = usii
                slot[i] = usii_triad[i, 3]
                slotm = slotm + slot[i]
            if star_slotsum != slotm:
                star_slotsum = slotm
            rdiff = cadac_matmul(triad_meas, cadac_inverse(triad_true))
            uric[0] = rdiff[2, 1]
            uric[1] = rdiff[0, 2]
            uric[2] = rdiff[1, 0]

        store.set("URIC", uric)
        store.set("mstar", mstar)
        store.set("starfix_epoch", starfix_epoch)
        store.set("star_acq", star_acq)
        store.set("star_slotsum", star_slotsum)
        store.set("star_volume", star_volume)

import math

import numpy as np

from cadac.constants import REARTH
from cadac.kernel.state import Field
from cadac.math.frames import angle
from cadac.math.wgs84 import GM

LARGE = 1e10

# Yuma Almanac Week 787 (21 Sep 2014): right ascension (rad), argument of latitude (rad)
SV_INIT = np.array(
    [
        [5.63, -1.600],
        [5.63, 2.115],
        [5.63, -2.309],
        [5.63, 0.319],
        [0.40, 1.063],
        [0.40, -1.342],
        [0.40, 0.543],
        [0.40, 2.874],
        [1.45, 1.705],
        [1.45, -2.841],
        [1.45, -2.321],
        [1.45, -0.640],
        [2.45, 1.941],
        [2.45, -0.147],
        [2.45, 1.690],
        [2.45, 0.409],
        [3.48, -0.571],
        [3.48, -2.988],
        [3.48, 0.858],
        [3.48, 2.705],
        [4.59, -0.7180],
        [4.59, 2.666],
        [4.59, -2.977],
        [4.59, -0.2090],
    ],
    dtype=float,
)
rsi = 26560000
incl = 0.95986
wsi = math.sqrt(GM / rsi**3)


def gps_sv_init():
    return np.array(SV_INIT, dtype=float, copy=True), rsi, wsi, incl


def _univec3(vec):
    d = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
    if d == 0.0:
        return np.zeros(3)
    return np.array([vec[0] / d, vec[1] / d, vec[2] / d], dtype=float)


def gps_quadriga(
    sv_init_data,
    rsi,
    wsi,
    incl,
    almanac_time,
    del_rearth,
    time,
    sbii,
    mgps,
):
    sv_data = np.array(sv_init_data, dtype=float, copy=True)
    sbii = np.asarray(sbii, dtype=float)
    gdop = LARGE
    ssii_quad = np.zeros((4, 4))
    vsii_quad = np.zeros((4, 3))

    sv_data[:, 1] = sv_data[:, 1] + (almanac_time + time) * wsi

    sin_incl = math.sin(incl)
    cos_incl = math.cos(incl)
    ssii = np.zeros((24, 4))
    for i in range(24):
        ra = sv_data[i, 0]
        arg = sv_data[i, 1]
        ssii[i, 0] = rsi * (
            math.cos(ra) * math.cos(arg) - math.sin(ra) * math.sin(arg) * cos_incl
        )
        ssii[i, 1] = rsi * (
            math.sin(ra) * math.cos(arg) + math.cos(ra) * math.sin(arg) * cos_incl
        )
        ssii[i, 2] = rsi * math.sin(arg) * sin_incl
        ssii[i, 3] = 0.0

    visible_count = 0
    epsilon = math.acos((REARTH + del_rearth) / rsi)
    dbi = math.sqrt(sbii[0] * sbii[0] + sbii[1] * sbii[1] + sbii[2] * sbii[2])
    for i in range(24):
        ssii_i = ssii[i, :3]
        delta = angle(ssii_i, sbii)
        rmin = (REARTH + del_rearth) / math.cos(delta - epsilon)
        if delta < epsilon:
            ssii[i, 3] = i + 1
            visible_count += 1
        elif rmin > 0 and rmin < dbi:
            ssii[i, 3] = i + 1
            visible_count += 1

    if visible_count < 4:
        return ssii_quad, vsii_quad, gdop, 1

    ssii_vis = np.zeros((visible_count, 4))
    k = 0
    for i in range(24):
        if ssii[i, 3] > 0:
            ssii_vis[k] = ssii[i]
            k += 1

    nm3 = visible_count - 3
    nm2 = visible_count - 2
    nm1 = visible_count - 1
    quad = [0, 0, 0, 0]
    for i1 in range(nm3):
        for i2 in range(i1 + 1, nm2):
            for i3 in range(i2 + 1, nm1):
                for i4 in range(i3 + 1, visible_count):
                    ssii1 = ssii_vis[i1, :3]
                    ssii2 = ssii_vis[i2, :3]
                    ssii3 = ssii_vis[i3, :3]
                    ssii4 = ssii_vis[i4, :3]
                    uni1 = _univec3(sbii - ssii1)
                    uni2 = _univec3(sbii - ssii2)
                    uni3 = _univec3(sbii - ssii3)
                    uni4 = _univec3(sbii - ssii4)
                    hgps = np.ones((4, 4))
                    hgps[0, :3] = uni1
                    hgps[1, :3] = uni2
                    hgps[2, :3] = uni3
                    hgps[3, :3] = uni4
                    cov = np.linalg.inv(hgps @ hgps.T)
                    gdop_local = math.sqrt(
                        cov[0, 0] + cov[1, 1] + cov[2, 2] + cov[3, 3]
                    )
                    if gdop_local < gdop:
                        gdop = gdop_local
                        quad[0] = i1
                        quad[1] = i2
                        quad[2] = i3
                        quad[3] = i4

    for m in range(4):
        ssii_quad[m] = ssii_vis[quad[m]]

    vel = rsi * wsi
    for m in range(4):
        ii = int(ssii_quad[m, 3]) - 1
        ra = sv_data[ii, 0]
        arg = sv_data[ii, 1]
        vsii_quad[m, 0] = vel * (
            -math.sin(arg) * math.cos(ra) - math.cos(arg) * math.sin(ra) * cos_incl
        )
        vsii_quad[m, 1] = vel * (
            -math.sin(arg) * math.sin(ra) + math.cos(arg) * math.cos(ra) * cos_incl
        )
        vsii_quad[m, 2] = vel * (math.cos(arg) * sin_incl)

    return ssii_quad, vsii_quad, gdop, mgps


class Rocket6Gps:
    name = "gps"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        zeros33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        plot = ("plot",)
        scrn_plot = ("scrn", "plot")
        for field in (
            Field("mgps", 0, "int", "data", "gps"),
            Field("almanac_time", 0.0, "real", "data", "gps"),
            Field("del_rearth", 0.0, "real", "data", "gps"),
            Field("gdop", 0.0, "real", "diag", "gps"),
            Field("gps_acqtime", 0.0, "real", "data", "gps"),
            Field("gps_step", 0.0, "real", "data", "gps"),
            Field("gps_epoch", 0.0, "real", "save", "gps"),
            Field("gps_acq", 0, "int", "save", "gps"),
            Field("ucfreq_noise", 0.0, "real", "data", "gps"),
            Field("ucbias_error", 0.0, "real", "data", "gps", scrn_plot),
            Field("ucfreq_error", 0.0, "real", "diag", "gps", scrn_plot),
            Field("ucfreqm", 0.0, "real", "save", "gps"),
            Field("pr1_bias", 0.0, "real", "data", "gps"),
            Field("pr2_bias", 0.0, "real", "data", "gps"),
            Field("pr3_bias", 0.0, "real", "data", "gps"),
            Field("pr4_bias", 0.0, "real", "data", "gps"),
            Field("pr1_noise", 0.0, "real", "data", "gps"),
            Field("pr2_noise", 0.0, "real", "data", "gps"),
            Field("pr3_noise", 0.0, "real", "data", "gps"),
            Field("pr4_noise", 0.0, "real", "data", "gps"),
            Field("dr1_noise", 0.0, "real", "data", "gps"),
            Field("dr2_noise", 0.0, "real", "data", "gps"),
            Field("dr3_noise", 0.0, "real", "data", "gps"),
            Field("dr4_noise", 0.0, "real", "data", "gps"),
            Field("slotsum", 0.0, "real", "save", "gps"),
            Field("lon1", 0.0, "real", "diag", "gps"),
            Field("lat1", 0.0, "real", "diag", "gps"),
            Field("alt1", 0.0, "real", "diag", "gps"),
            Field("lon2", 0.0, "real", "diag", "gps"),
            Field("lat2", 0.0, "real", "diag", "gps"),
            Field("alt2", 0.0, "real", "diag", "gps"),
            Field("lon3", 0.0, "real", "diag", "gps"),
            Field("lat3", 0.0, "real", "diag", "gps"),
            Field("alt3", 0.0, "real", "diag", "gps"),
            Field("lon4", 0.0, "real", "diag", "gps"),
            Field("lat4", 0.0, "real", "diag", "gps"),
            Field("alt4", 0.0, "real", "diag", "gps"),
            Field("dum_alt", 0.0, "real", "diag", "gps"),
            Field("uctime_cor", 0.0, "real", "data", "gps"),
            Field("ppos", 0.0, "real", "data", "gps"),
            Field("pvel", 0.0, "real", "data", "gps"),
            Field("pclockb", 0.0, "real", "data", "gps"),
            Field("pclockf", 0.0, "real", "data", "gps"),
            Field("qpos", 0.0, "real", "data", "gps"),
            Field("qvel", 0.0, "real", "data", "gps"),
            Field("qclockb", 0.0, "real", "data", "gps"),
            Field("qclockf", 0.0, "real", "data", "gps"),
            Field("rpos", 0.0, "real", "data", "gps"),
            Field("rvel", 0.0, "real", "data", "gps"),
            Field("factp", 0.0, "real", "data", "gps"),
            Field("factq", 0.0, "real", "data", "gps"),
            Field("factr", 0.0, "real", "data", "gps"),
            Field("SXH", zeros3, "vec", "out", "gps", plot),
            Field("VXH", zeros3, "vec", "out", "gps", plot),
            Field("CXH", zeros3, "vec", "save", "gps", plot),
            Field("PP1", zeros33, "mat", "save", "gps"),
            Field("PP2", zeros33, "mat", "save", "gps"),
            Field("PP3", zeros33, "mat", "save", "gps"),
            Field("PP4", zeros33, "mat", "save", "gps"),
            Field("PP5", zeros33, "mat", "save", "gps"),
            Field("PP6", zeros33, "mat", "save", "gps"),
            Field("PP7", zeros33, "mat", "save", "gps"),
            Field("PP8", zeros33, "mat", "save", "gps"),
            Field("std_pos", 0.0, "real", "diag", "gps", plot),
            Field("std_vel", 0.0, "real", "diag", "gps", plot),
            Field("std_ucbias", 0.0, "real", "diag", "gps", plot),
            Field("c2_pos_meas", 0.0, "real", "diag", "gps", plot),
            Field("c2_vel_meas", 0.0, "real", "diag", "gps", plot),
            Field("state_pos", 0.0, "real", "diag", "gps", plot),
            Field("state_vel", 0.0, "real", "diag", "gps", plot),
            Field("c2_range_err", 0.0, "real", "diag", "gps", plot),
            Field("c2_delta_err", 0.0, "real", "diag", "gps", plot),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        pass

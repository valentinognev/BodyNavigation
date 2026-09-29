"""HYPER6 GPS / Kalman filter — port of Hyper::gps (gps.cpp)."""

import math

import numpy as np

from cadac.constants import REARTH
from cadac.kernel.state import Field
from cadac.math.frames import angle, cadac_inverse, cadac_matmul, skew
from cadac.math.wgs84 import GM

LARGE = 1e10

# Week 224 circular-orbit SV init (Hyper::gps_sv_init): RA (rad), arg of latitude (rad)
SV_INIT = np.array(
    [
        [0.9947, 1.0371],  # A-plane
        [0.9947, 2.7284],
        [0.9947, -2.9855],
        [0.9947, -0.9112],
        [1.9197, 1.6417],  # B-plane
        [1.9197, -2.2572],
        [1.9197, -0.7270],
        [1.9197, -0.1626],
        [2.9668, 0.5080],  # C-plane
        [2.9668, 2.1743],
        [2.9668, 2.7649],
        [2.9668, -1.8197],
        [4.0140, 0.9408],  # D-plane
        [4.0140, 2.4279],
        [4.0140, -3.1153],
        [4.0140, -0.7710],
        [5.0611, 1.3270],  # E-plane
        [5.0611, 2.1057],
        [5.0611, -2.5338],
        [5.0611, -0.2883],
        [6.1082, 0.3998],  # F-plane
        [6.1082, 2.3427],
        [6.1082, -1.8972],
        [6.1082, -1.4035],
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
                    cov = cadac_inverse(cadac_matmul(hgps, hgps.T.copy()))
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
            -math.cos(ra) * math.sin(arg) - math.sin(ra) * math.cos(arg) * cos_incl
        )
        vsii_quad[m, 1] = vel * (
            -math.sin(ra) * math.sin(arg) + math.cos(ra) * math.cos(arg) * cos_incl
        )
        vsii_quad[m, 2] = vel * (math.cos(arg) * sin_incl)

    return ssii_quad, vsii_quad, gdop, mgps


class Hyper6Gps:
    name = "gps"

    def __init__(self):
        self.PP = np.zeros((8, 8))
        self.QQ = np.zeros((8, 8))
        self.RR = np.zeros((8, 8))
        self.PHI = np.zeros((8, 8))
        self.FF = np.zeros((8, 8))
        self.sv_init_data, self.rsi, self.wsi, self.incl = gps_sv_init()

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = (0.0, 0.0, 0.0)
        zeros33 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        plot = ("plot",)
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
            Field("ucbias_error", 0.0, "real", "data", "gps"),
            Field("ucfreq_error", 0.0, "real", "diag", "gps"),
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
            Field("VXH", zeros3, "vec", "out", "gps"),
            Field("CXH", zeros3, "vec", "save", "gps"),
            Field("PP1", zeros33, "mat", "save", "gps"),
            Field("PP2", zeros33, "mat", "save", "gps"),
            Field("PP3", zeros33, "mat", "save", "gps"),
            Field("PP4", zeros33, "mat", "save", "gps"),
            Field("PP5", zeros33, "mat", "save", "gps"),
            Field("PP6", zeros33, "mat", "save", "gps"),
            Field("PP7", zeros33, "mat", "save", "gps"),
            Field("PP8", zeros33, "mat", "save", "gps"),
            Field("std_pos", 0.0, "real", "diag", "gps", plot),
            Field("std_vel", 0.0, "real", "diag", "gps"),
            Field("std_ucbias", 0.0, "real", "diag", "gps", plot),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mgps = store.get("mgps")
        if mgps not in (0, 1, 2, 3):
            raise ValueError(f"unknown mgps {mgps}")
        if mgps == 0:
            return

        int_step = ctx.int_step
        time = store.get("time")
        almanac_time = store.get("almanac_time")
        del_rearth = store.get("del_rearth")
        gps_acqtime = store.get("gps_acqtime")
        gps_step = store.get("gps_step")
        ucfreq_noise = store.get("ucfreq_noise")
        ucbias_error = store.get("ucbias_error")
        pr_bias = np.array(
            [
                store.get("pr1_bias"),
                store.get("pr2_bias"),
                store.get("pr3_bias"),
                store.get("pr4_bias"),
            ],
            dtype=float,
        )
        pr_noise = np.array(
            [
                store.get("pr1_noise"),
                store.get("pr2_noise"),
                store.get("pr3_noise"),
                store.get("pr4_noise"),
            ],
            dtype=float,
        )
        dr_noise = np.array(
            [
                store.get("dr1_noise"),
                store.get("dr2_noise"),
                store.get("dr3_noise"),
                store.get("dr4_noise"),
            ],
            dtype=float,
        )
        uctime_cor = store.get("uctime_cor")
        ppos = store.get("ppos")
        pvel = store.get("pvel")
        pclockb = store.get("pclockb")
        pclockf = store.get("pclockf")
        qpos = store.get("qpos")
        qvel = store.get("qvel")
        qclockb = store.get("qclockb")
        qclockf = store.get("qclockf")
        rpos = store.get("rpos")
        rvel = store.get("rvel")
        factp = store.get("factp")
        factq = store.get("factq")
        factr = store.get("factr")
        gps_epoch = store.get("gps_epoch")
        gps_acq = store.get("gps_acq")
        ucfreqm = store.get("ucfreqm")
        slotsum = store.get("slotsum")
        sxh = np.asarray(store.get("SXH"), dtype=float).copy()
        vxh = np.asarray(store.get("VXH"), dtype=float).copy()
        cxh = np.asarray(store.get("CXH"), dtype=float).copy()

        gdop = 0.0
        ucfreq_error = 0.0
        std_pos = 0.0
        std_vel = 0.0
        std_ucbias = 0.0

        if mgps == 1:
            self.sv_init_data, self.rsi, self.wsi, self.incl = gps_sv_init()
            for i in range(3):
                self.PP[i, i] = (ppos * (1.0 + factp)) ** 2
                self.PP[i + 3, i + 3] = (pvel * (1.0 + factp)) ** 2
            self.PP[6, 6] = (pclockb * (1.0 + factp)) ** 2
            self.PP[7, 7] = (pclockf * (1.0 + factp)) ** 2
            self.QQ[:, :] = 0.0
            for i in range(3):
                self.QQ[i, i] = (qpos * (1.0 + factq)) ** 2
                self.QQ[i + 3, i + 3] = (qvel * (1.0 + factq)) ** 2
            self.QQ[6, 6] = (qclockb * (1.0 + factq)) ** 2
            self.QQ[7, 7] = (qclockf * (1.0 + factq)) ** 2
            self.RR[:, :] = 0.0
            for i in range(4):
                self.RR[i, i] = (rpos * (1.0 + factr)) ** 2
                self.RR[i + 4, i + 4] = (rvel * (1.0 + factr)) ** 2
            self.FF[:, :] = 0.0
            self.FF[0, 3] = 1.0
            self.FF[1, 4] = 1.0
            self.FF[2, 5] = 1.0
            self.FF[6, 7] = 1.0
            self.FF[7, 7] = -1.0 / uctime_cor
            self.PHI = (
                np.eye(8)
                + self.FF * int_step
                + cadac_matmul(self.FF, self.FF) * (int_step * int_step / 2.0)
            )
            gps_acq = 1
            gps_epoch = time
            mgps = 2

        if mgps == 2:
            if gps_acq:
                dtime_gps = gps_acqtime
            else:
                dtime_gps = gps_step
            time_gps = time - gps_epoch
            if time_gps >= dtime_gps:
                mgps = 3
            ucfreq_error = ucfreq_noise
            ucbias_error = ucbias_error + (ucfreq_error + ucfreqm) * (int_step / 2.0)
            ucfreqm = ucfreq_error
            mid = self.PP + self.QQ * (int_step / 2.0)
            self.PP = cadac_matmul(
                cadac_matmul(self.PHI, mid), self.PHI.T.copy()
            ) + self.QQ * (int_step / 2.0)
            std_pos = math.sqrt(self.PP[0, 0])
            std_vel = math.sqrt(self.PP[3, 3])
            std_ucbias = math.sqrt(self.PP[6, 6])

        if mgps == 3:
            gps_acq = 0
            gps_epoch = time
            sbii = np.asarray(store.get("SBII"), dtype=float)
            vbii = np.asarray(store.get("VBII"), dtype=float)
            wbii = np.asarray(store.get("WBII"), dtype=float)
            sbiic = np.asarray(store.get("SBIIC"), dtype=float)
            vbiic = np.asarray(store.get("VBIIC"), dtype=float)
            wbici = np.asarray(store.get("WBICI"), dtype=float)
            ssii_quad, vsii_quad, gdop, mgps = gps_quadriga(
                self.sv_init_data,
                self.rsi,
                self.wsi,
                self.incl,
                almanac_time,
                del_rearth,
                time,
                sbii,
                mgps,
            )
            zz = np.zeros(8)
            hh = np.zeros((8, 8))
            slotm = 0.0
            wbii_skew = skew(wbii)
            wbici_skew = skew(wbici)
            for i in range(4):
                ssii = ssii_quad[i, :3]
                ssbi = ssii - sbii
                dsb = math.sqrt(ssbi[0] * ssbi[0] + ssbi[1] * ssbi[1] + ssbi[2] * ssbi[2])
                dsb_meas = dsb + pr_bias[i] + pr_noise[i] + ucbias_error
                vsii = vsii_quad[i]
                vsbi = vsii - vbii - cadac_matmul(wbii_skew, ssbi)
                ussbi = ssbi * (1.0 / dsb)
                dvsb = float(vsbi[0] * ussbi[0] + vsbi[1] * ussbi[1] + vsbi[2] * ussbi[2])
                dvsb_meas = dvsb + dr_noise[i] + ucfreq_error
                ssbic = ssii - sbiic
                dsbc = math.sqrt(
                    ssbic[0] * ssbic[0] + ssbic[1] * ssbic[1] + ssbic[2] * ssbic[2]
                )
                vsbic = vsii - vbiic - cadac_matmul(wbici_skew, ssbic)
                ussbic = ssbic * (1.0 / dsb)
                dvsbc = float(
                    vsbic[0] * ussbic[0] + vsbic[1] * ussbic[1] + vsbic[2] * ussbic[2]
                )
                zz[i] = dsb_meas - dsbc
                zz[i + 4] = dvsb_meas - dvsbc
                hh[i, :3] = ussbi
                hh[i + 4, 3:6] = ussbi * gps_step
                hh[i, 6] = 1.0
                hh[i + 4, 7] = gps_step
                slotm = slotm + ssii_quad[i, 3]
            if slotsum != slotm:
                slotsum = slotm
            innov = cadac_matmul(cadac_matmul(hh, self.PP), hh.T.copy()) + self.RR
            kk = cadac_matmul(
                cadac_matmul(self.PP, hh.T.copy()), cadac_inverse(innov)
            )
            xh = cadac_matmul(kk, zz)
            self.PP = cadac_matmul(np.eye(8) - cadac_matmul(kk, hh), self.PP)
            ucbias_error = ucbias_error - xh[6]
            sxh = np.array([xh[0], xh[1], xh[2]], dtype=float)
            vxh = np.array([xh[3], xh[4], xh[5]], dtype=float)
            cxh[0] = xh[6]
            cxh[1] = xh[7]

        store.set("mgps", mgps)
        store.set("gps_epoch", gps_epoch)
        store.set("gps_acq", gps_acq)
        store.set("ucbias_error", ucbias_error)
        store.set("ucfreqm", ucfreqm)
        store.set("slotsum", slotsum)
        store.set("SXH", sxh)
        store.set("VXH", vxh)
        store.set("CXH", cxh)
        store.set("gdop", gdop)
        store.set("ucfreq_error", ucfreq_error)
        store.set("std_pos", std_pos)
        store.set("std_vel", std_vel)
        store.set("std_ucbias", std_ucbias)

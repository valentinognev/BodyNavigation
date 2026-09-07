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


class Rocket6Gps:
    name = "gps"

    def __init__(self):
        self.PP = np.zeros((8, 8))
        self.PHI = np.zeros((8, 8))
        self.FF = np.zeros((8, 8))
        self.sv_init_data, self.rsi, self.wsi, self.incl = gps_sv_init()

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
        c2_pos_meas = store.get("c2_pos_meas")
        c2_vel_meas = store.get("c2_vel_meas")
        state_pos = store.get("state_pos")
        state_vel = store.get("state_vel")
        c2_range_err = store.get("c2_range_err")
        c2_delta_err = store.get("c2_delta_err")
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
            self.FF[:, :] = 0.0
            self.FF[0, 3] = 1.0
            self.FF[1, 4] = 1.0
            self.FF[2, 5] = 1.0
            self.FF[6, 7] = 1.0
            self.FF[7, 7] = -1.0 / uctime_cor
            self.PHI = (
                np.eye(8)
                + self.FF * int_step
                + (self.FF @ self.FF) * (int_step * int_step / 2.0)
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
            qq = np.zeros((8, 8))
            for i in range(3):
                qq[i, i] = (qpos * (1.0 + factq)) ** 2
                qq[i + 3, i + 3] = (qvel * (1.0 + factq)) ** 2
            qq[6, 6] = (qclockb * (1.0 + factq)) ** 2
            qq[7, 7] = (qclockf * (1.0 + factq)) ** 2
            self.PP = (
                self.PHI @ (self.PP + qq * (int_step / 2.0)) @ self.PHI.T
                + qq * (int_step / 2.0)
            )
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
            wbii_skew = _skew(wbii)
            wbici_skew = _skew(wbici)
            for i in range(4):
                ssii = ssii_quad[i, :3]
                ssbi = ssii - sbii
                dsb = math.sqrt(ssbi[0] * ssbi[0] + ssbi[1] * ssbi[1] + ssbi[2] * ssbi[2])
                dsb_meas = dsb + pr_bias[i] + pr_noise[i] + ucbias_error
                if i == 0:
                    c2_range_err = dsb_meas - dsb
                vsii = vsii_quad[i]
                vsbi = vsii - vbii - wbii_skew @ ssbi
                ussbi = ssbi * (1.0 / dsb)
                dvsb = float(vsbi[0] * ussbi[0] + vsbi[1] * ussbi[1] + vsbi[2] * ussbi[2])
                dvsb_meas = dvsb + dr_noise[i] + ucfreq_error
                if i == 0:
                    c2_delta_err = dvsb_meas - dvsb
                ssbic = ssii - sbiic
                dsbc = math.sqrt(
                    ssbic[0] * ssbic[0] + ssbic[1] * ssbic[1] + ssbic[2] * ssbic[2]
                )
                vsbic = vsii - vbiic - wbici_skew @ ssbic
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
            rr = np.zeros((8, 8))
            for i in range(4):
                rr[i, i] = (rpos * (1.0 + factr)) ** 2
                rr[i + 4, i + 4] = (rvel * (1.0 + factr)) ** 2
            kk = self.PP @ hh.T @ np.linalg.inv(hh @ self.PP @ hh.T + rr)
            xh = kk @ zz
            self.PP = (np.eye(8) - kk @ hh) @ self.PP
            ucbias_error = ucbias_error - xh[6]
            c2_pos_meas = float(zz[0])
            c2_vel_meas = float(zz[4])
            sxh = np.array([xh[0], xh[1], xh[2]], dtype=float)
            vxh = np.array([xh[3], xh[4], xh[5]], dtype=float)
            cxh[0] = xh[6]
            cxh[1] = xh[7]
            state_pos = math.sqrt(sxh[0] * sxh[0] + sxh[1] * sxh[1] + sxh[2] * sxh[2])
            state_vel = math.sqrt(vxh[0] * vxh[0] + vxh[1] * vxh[1] + vxh[2] * vxh[2])

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
        store.set("c2_pos_meas", c2_pos_meas)
        store.set("c2_vel_meas", c2_vel_meas)
        store.set("state_pos", state_pos)
        store.set("state_vel", state_vel)
        store.set("c2_range_err", c2_range_err)
        store.set("c2_delta_err", c2_delta_err)

"""CRUISE5 GPS / 17-state filter — Fortran MODULE.FOR S2I / S2 (MGPS 0/1/2)."""

from __future__ import annotations

import math

import numpy as np

from cadac.constants import AGRAV, DEG, REARTH
from cadac.kernel.state import Field
from cadac.math.frames import cadac_inverse, cadac_matmul

RGPS = 20183.0e3
_ZEROS3 = (0.0, 0.0, 0.0)
_EYE3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _matcar(magnitude: float, azimuth: float, elevation: float) -> np.ndarray:
    """Fortran UTL MATCAR: cartesian from polar (mag, az, el)."""
    cel = math.cos(elevation)
    return np.array(
        [
            magnitude * cel * math.cos(azimuth),
            magnitude * cel * math.sin(azimuth),
            -magnitude * math.sin(elevation),
        ],
        dtype=float,
    )


def _univec(vec: np.ndarray) -> np.ndarray:
    """Fortran VECUVC."""
    d = math.sqrt(float(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]))
    if d == 0.0:
        return np.zeros(3, dtype=float)
    return np.array([vec[0] / d, vec[1] / d, vec[2] / d], dtype=float)


def _mattrf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fortran MATTRF: A * B * A^T."""
    return cadac_matmul(cadac_matmul(a, b), a.T.copy())


def _cadelp(aa: np.ndarray) -> tuple[float, float, float]:
    """Fortran CADELP — major/minor eigenvalues and major-axis angle."""
    a11 = float(aa[0, 0])
    a22 = float(aa[1, 1])
    a12 = float(aa[0, 1])
    a1133 = a11 + a22
    dum1 = a1133 * a1133 - 4.0 * (a11 * a22 - a12 * a12)
    dum2 = math.sqrt(dum1) if dum1 >= 0.0 else 0.0
    ama = (a1133 + dum2) / 2.0
    ami = (a1133 - dum2) / 2.0
    phi = 0.0
    if ama == ami:
        return ama, ami, phi
    if a11 - ama != 0.0:
        dum = -a12 / (a11 - ama)
        ak1 = math.sqrt(1.0 / (1.0 + dum * dum))
        dum = dum * ak1
        if abs(dum) > 1.0:
            dum = math.copysign(1.0, dum)
        phi = math.acos(dum)
    else:
        dum1 = -a12 / (a22 - ama)
        ak1 = math.sqrt(1.0 / (1.0 + dum1 * dum1))
        if abs(ak1) > 1.0:
            ak1 = math.copysign(1.0, ak1)
        phi = math.acos(ak1)
    return ama, ami, phi


class Cruise5Gps:
    """Zipfel CRUISE5 flat-Earth GPS (Fortran ``S2I`` / ``S2``)."""

    name = "gps"

    def __init__(self):
        self.pp = np.zeros((17, 17), dtype=float)
        self.qq = np.zeros((17, 17), dtype=float)
        self.rr = np.zeros((8, 8), dtype=float)
        self.ee = np.eye(17, dtype=float)
        self.xh = np.zeros(17, dtype=float)
        self.sge = np.zeros((3, 4), dtype=float)
        self.cfreqm = 0.0
        self.tgps = 0.0

    def define(self, vehicle):
        store = vehicle.store
        for field in (
            Field("mgps", 0, "int", "data", "gps"),
            Field("ms2prt", 0, "int", "data", "gps"),
            Field("azgex1", 0.0, "real", "data", "gps"),
            Field("azgex2", 0.0, "real", "data", "gps"),
            Field("azgex3", 0.0, "real", "data", "gps"),
            Field("azgex4", 0.0, "real", "data", "gps"),
            Field("elgex1", 0.0, "real", "data", "gps"),
            Field("elgex2", 0.0, "real", "data", "gps"),
            Field("elgex3", 0.0, "real", "data", "gps"),
            Field("elgex4", 0.0, "real", "data", "gps"),
            Field("pspos", 0.0, "real", "data", "gps"),
            Field("psvel", 0.0, "real", "data", "gps"),
            Field("pstil", 0.0, "real", "data", "gps"),
            Field("psacc", 0.0, "real", "data", "gps"),
            Field("psgyr", 0.0, "real", "data", "gps"),
            Field("pscbi", 0.0, "real", "data", "gps"),
            Field("pscfr", 0.0, "real", "data", "gps"),
            Field("frapi", 0.0, "real", "data", "gps"),
            Field("frapa", 0.0, "real", "data", "gps"),
            Field("frapg", 0.0, "real", "data", "gps"),
            Field("frapc", 0.0, "real", "data", "gps"),
            Field("qspos", 0.0, "real", "data", "gps"),
            Field("qsvel", 0.0, "real", "data", "gps"),
            Field("qstil", 0.0, "real", "data", "gps"),
            Field("qsacc", 0.0, "real", "data", "gps"),
            Field("qsgyr", 0.0, "real", "data", "gps"),
            Field("qscbi", 0.0, "real", "data", "gps"),
            Field("qscfr", 0.0, "real", "data", "gps"),
            Field("fraq", 0.0, "real", "data", "gps"),
            Field("rspos", 0.0, "real", "data", "gps"),
            Field("rsvel", 0.0, "real", "data", "gps"),
            Field("frar", 0.0, "real", "data", "gps"),
            Field("tc", 0.0, "real", "data", "gps"),
            Field("brpat1", 0.0, "real", "data", "gps"),
            Field("brpat2", 0.0, "real", "data", "gps"),
            Field("brpat3", 0.0, "real", "data", "gps"),
            Field("brpat4", 0.0, "real", "data", "gps"),
            Field("rrrec1", 0.0, "real", "data", "gps"),
            Field("rrrec2", 0.0, "real", "data", "gps"),
            Field("rrrec3", 0.0, "real", "data", "gps"),
            Field("rrrec4", 0.0, "real", "data", "gps"),
            Field("rddyn1", 0.0, "real", "data", "gps"),
            Field("rddyn2", 0.0, "real", "data", "gps"),
            Field("rddyn3", 0.0, "real", "data", "gps"),
            Field("rddyn4", 0.0, "real", "data", "gps"),
            Field("cbias", 0.0, "real", "data", "gps"),
            Field("cfreq", 0.0, "real", "data", "gps"),
            Field("dtimgps", 0.0, "real", "data", "gps"),
            Field("elma", 0.0, "real", "diag", "gps"),
            Field("elmi", 0.0, "real", "diag", "gps"),
            Field("elphix", 0.0, "real", "diag", "gps"),
            Field("gucbias", 0.0, "real", "diag", "gps"),
            Field("gucfreq", 0.0, "real", "diag", "gps"),
            Field("ucfreq", 0.0, "real", "diag", "gps"),
            Field("gdop", 0.0, "real", "diag", "gps"),
            Field("GUSTTCL", _ZEROS3, "vec", "out", "gps"),
            Field("GUVBEL", _ZEROS3, "vec", "out", "gps"),
            Field("GURECEL", _ZEROS3, "vec", "out", "gps"),
            Field("GUFSPB", _ZEROS3, "vec", "out", "gps"),
            Field("GUWBEB", _ZEROS3, "vec", "out", "gps"),
            # Flat-Earth / INS handshake inputs (Task 75 owns real INS).
            Field("SBEL", _ZEROS3, "vec", "state", "gps"),
            Field("VBEL", _ZEROS3, "vec", "state", "gps"),
            Field("VBELC", _ZEROS3, "vec", "out", "gps"),
            Field("ESTTC", _ZEROS3, "vec", "out", "gps"),
            Field("FSPCB", _ZEROS3, "vec", "out", "gps"),
            Field("TBLC", _EYE3, "mat", "out", "gps"),
            Field("TVL", _EYE3, "mat", "out", "gps"),
            Field("tanlat", 0.0, "real", "data", "gps"),
        ):
            if field.name not in store:
                store.define(field)
        if "time" not in store:
            store.define(Field("time", 0.0, "real", "exec", "environment"))

    def initialize(self, vehicle, ctx):
        """Fortran S2I — satellite geometry + PP/QQ/RR init."""
        store = vehicle.store
        azgex = [
            float(store.get("azgex1")),
            float(store.get("azgex2")),
            float(store.get("azgex3")),
            float(store.get("azgex4")),
        ]
        elgex = [
            float(store.get("elgex1")),
            float(store.get("elgex2")),
            float(store.get("elgex3")),
            float(store.get("elgex4")),
        ]
        pspos = float(store.get("pspos"))
        psvel = float(store.get("psvel"))
        pstil = float(store.get("pstil"))
        psacc = float(store.get("psacc"))
        psgyr = float(store.get("psgyr"))
        pscbi = float(store.get("pscbi"))
        pscfr = float(store.get("pscfr"))
        frapi = float(store.get("frapi"))
        frapa = float(store.get("frapa"))
        frapg = float(store.get("frapg"))
        frapc = float(store.get("frapc"))
        qspos = float(store.get("qspos"))
        qsvel = float(store.get("qsvel"))
        qstil = float(store.get("qstil"))
        qsacc = float(store.get("qsacc"))
        qsgyr = float(store.get("qsgyr"))
        qscbi = float(store.get("qscbi"))
        qscfr = float(store.get("qscfr"))
        fraq = float(store.get("fraq"))
        rspos = float(store.get("rspos"))
        rsvel = float(store.get("rsvel"))
        frar = float(store.get("frar"))

        self.ee = np.eye(17, dtype=float)
        self.xh = np.zeros(17, dtype=float)
        self.cfreqm = 0.0
        self.tgps = 0.0

        hgps = np.zeros((4, 4), dtype=float)
        for i in range(4):
            el = elgex[i] / DEG
            az = azgex[i] / DEG
            angl = 1.570796 + el
            dum = REARTH * math.cos(angl)
            dge = dum + math.sqrt(dum * dum + RGPS * RGPS - REARTH * REARTH)
            sgel = _matcar(dge, az, el)
            self.sge[:, i] = sgel
            ugel = _univec(sgel)
            hgps[i, :3] = ugel
            hgps[i, 3] = 1.0

        hgpst = hgps.T.copy()
        dum44 = cadac_matmul(hgpst, hgps)
        dum44i = cadac_inverse(dum44)
        tdop2 = float(dum44i[3, 3])
        vdop2 = float(dum44i[2, 2])
        hdop2 = float(dum44i[0, 0] + dum44i[1, 1])
        pdop2 = vdop2 + hdop2
        gdop2 = pdop2 + tdop2
        store.set("gdop", math.sqrt(gdop2))

        self.pp[:, :] = 0.0
        for i in range(3):
            self.pp[i, i] = ((1.0 + frapi) * pspos) ** 2
            self.pp[i + 3, i + 3] = ((1.0 + frapi) * psvel) ** 2
            self.pp[i + 6, i + 6] = ((1.0 + frapi) * pstil) ** 2
            self.pp[i + 9, i + 9] = ((1.0 + frapa) * psacc) ** 2
            self.pp[i + 12, i + 12] = ((1.0 + frapg) * psgyr) ** 2
        self.pp[15, 15] = ((1.0 + frapc) * pscbi) ** 2
        self.pp[16, 16] = ((1.0 + frapc) * pscfr) ** 2

        self.qq[:, :] = 0.0
        for i in range(3):
            self.qq[i, i] = ((1.0 + fraq) * qspos) ** 2
            self.qq[i + 3, i + 3] = ((1.0 + fraq) * qsvel) ** 2
            self.qq[i + 6, i + 6] = ((1.0 + fraq) * qstil) ** 2
            self.qq[i + 9, i + 9] = ((1.0 + fraq) * qsacc) ** 2
            self.qq[i + 12, i + 12] = ((1.0 + fraq) * qsgyr) ** 2
        self.qq[15, 15] = ((1.0 + fraq) * qscbi) ** 2
        self.qq[16, 16] = ((1.0 + fraq) * qscfr) ** 2

        self.rr[:, :] = 0.0
        for i in range(4):
            self.rr[i, i] = ((1.0 + frar) * rspos) ** 2
            self.rr[i + 4, i + 4] = ((1.0 + frar) * rsvel) ** 2

    def execute(self, vehicle, ctx):
        """Fortran S2 — filter extrapolate; MGPS≥1 meas; MGPS=2 Kalman→S4 handshake."""
        store = vehicle.store
        mgps = int(store.get("mgps"))
        if mgps not in (0, 1, 2):
            raise ValueError(f"unknown mgps {mgps}")

        der = float(ctx.int_step)
        time = float(store.get("time"))
        tc = float(store.get("tc"))
        dtimgps = float(store.get("dtimgps"))
        tanlat = float(store.get("tanlat"))
        cbias = float(store.get("cbias"))
        cfreq = float(store.get("cfreq"))
        gucfreq = float(store.get("gucfreq"))

        sbel = np.asarray(store.get("SBEL"), dtype=float).reshape(3).copy()
        vbel = np.asarray(store.get("VBEL"), dtype=float).reshape(3).copy()
        vbelc = np.asarray(store.get("VBELC"), dtype=float).reshape(3).copy()
        esttc = np.asarray(store.get("ESTTC"), dtype=float).reshape(3).copy()
        fspcb = np.asarray(store.get("FSPCB"), dtype=float).reshape(3).copy()
        tblc = np.asarray(store.get("TBLC"), dtype=float).reshape(3, 3).copy()
        tvl = np.asarray(store.get("TVL"), dtype=float).reshape(3, 3).copy()

        sbelc = esttc + sbel

        # Fundamental matrix FF (17×17)
        tlcb = tblc.T.copy()
        fsplc = cadac_matmul(tlcb, fspcb)
        ff = np.zeros((17, 17), dtype=float)
        ff[5, 2] = 2.0 * AGRAV / REARTH
        ff[6, 4] = 1.0 / REARTH
        ff[7, 3] = -1.0 / REARTH
        ff[8, 4] = -tanlat / REARTH
        ff[0, 3] = 1.0
        ff[1, 4] = 1.0
        ff[2, 5] = 1.0
        ff[3, 7] = -fsplc[2]
        ff[3, 8] = fsplc[1]
        ff[4, 6] = fsplc[2]
        ff[4, 8] = -fsplc[0]
        ff[5, 6] = -fsplc[1]
        ff[5, 7] = fsplc[0]
        for i in range(3):
            for j in range(3):
                ff[3 + i, 9 + j] = tlcb[i, j]
                ff[6 + i, 12 + j] = tlcb[i, j]
        ff[15, 16] = 1.0
        ff[16, 16] = -1.0 / tc if tc != 0.0 else 0.0

        # PHI = I + FF*dt + 0.5*(FF*dt)^2
        fdt = ff * der
        fdt2 = 0.5 * cadac_matmul(fdt, fdt)
        phi = self.ee + fdt + fdt2

        # State / covariance extrapolation
        self.xh = cadac_matmul(phi, self.xh)
        qdt = self.qq * (der / 2.0)
        mid = self.pp + qdt
        self.pp = _mattrf(phi, mid) + qdt

        # 1-sig diagnostics
        store.set("pspos", math.sqrt(float(self.pp[0, 0])))
        store.set("psvel", math.sqrt(float(self.pp[3, 3])))
        store.set("pstil", math.sqrt(float(self.pp[6, 6])))
        store.set("psacc", math.sqrt(float(self.pp[9, 9])))
        store.set("psgyr", math.sqrt(float(self.pp[12, 12])))
        store.set("pscbi", math.sqrt(float(self.pp[15, 15])))
        store.set("pscfr", math.sqrt(float(self.pp[16, 16])))

        covpl = self.pp[:3, :3].copy()
        covpv = _mattrf(tvl, covpl)
        cov23 = np.array(
            [
                [covpv[1, 1], covpv[1, 2]],
                [covpv[2, 1], covpv[2, 2]],
            ],
            dtype=float,
        )
        ama, ami, elphi = _cadelp(cov23)
        store.set("elma", math.sqrt(ama) if ama > 0.0 else 0.0)
        store.set("elmi", math.sqrt(ami) if ami > 0.0 else 0.0)
        store.set("elphix", elphi * DEG)

        # User clock dynamic model
        ucfreq = cfreq - gucfreq
        cbias = cbias + (ucfreq + self.cfreqm) * (der / 2.0)
        self.cfreqm = ucfreq
        store.set("ucfreq", ucfreq)
        store.set("cbias", cbias)

        if mgps == 0:
            store.set("mgps", mgps)
            return

        brpat = [
            float(store.get("brpat1")),
            float(store.get("brpat2")),
            float(store.get("brpat3")),
            float(store.get("brpat4")),
        ]
        rrrec = [
            float(store.get("rrrec1")),
            float(store.get("rrrec2")),
            float(store.get("rrrec3")),
            float(store.get("rrrec4")),
        ]
        rddyn = [
            float(store.get("rddyn1")),
            float(store.get("rddyn2")),
            float(store.get("rddyn3")),
            float(store.get("rddyn4")),
        ]

        hh = np.zeros((8, 17), dtype=float)
        zz = np.zeros(8, dtype=float)
        for i in range(4):
            sgel = self.sge[:, i]
            # True / measured range (double like Fortran DMAT*)
            dsgbl = sgel.astype(np.float64) - sbel.astype(np.float64)
            ddgb = math.sqrt(float(np.dot(dsgbl, dsgbl)))
            ddgbr = ddgb + brpat[i] + rrrec[i] + cbias
            sgbl = dsgbl.astype(float)
            ugbl = _univec(sgbl)
            dvbg = float(np.dot(vbel, ugbl))
            dvbgr = dvbg + rddyn[i] + ucfreq

            dsgblc = sgel.astype(np.float64) - sbelc.astype(np.float64)
            ddgbc = math.sqrt(float(np.dot(dsgblc, dsgblc)))
            sgblc = dsgblc.astype(float)
            ugblc = _univec(sgblc)
            dvbgc = float(np.dot(vbelc, ugblc))

            zz[i] = float(ddgbr - ddgbc)
            zz[i + 4] = dvbgr - dvbgc
            hh[i, :3] = ugblc
            hh[i + 4, 3:6] = ugblc * dtimgps
            hh[i, 15] = 1.0
            hh[i + 4, 16] = dtimgps

        if (time - self.tgps) >= dtimgps:
            mgps = 2
        if mgps < 2:
            store.set("mgps", mgps)
            return

        self.tgps = time

        # Kalman gain (double-precision path mirrors Fortran DMAT*)
        dhh = hh.astype(np.float64)
        dpp = self.pp.astype(np.float64)
        drr = self.rr.astype(np.float64)
        dht = dhh.T.copy()
        db178 = cadac_matmul(dpp, dht)
        db88 = cadac_matmul(dhh, db178)
        dby88 = db88 + drr
        db88i = cadac_inverse(dby88)
        dgk = cadac_matmul(db178, db88i)
        gk = dgk.astype(float)

        dx = cadac_matmul(gk, zz)
        self.xh = self.xh + dx
        self.pp = cadac_matmul(self.ee - cadac_matmul(gk, hh), self.pp)

        gusttcl = self.xh[0:3].copy()
        guvbel = self.xh[3:6].copy()
        gurecel = self.xh[6:9].copy()
        gufspl = self.xh[9:12].copy()
        guwbel = self.xh[12:15].copy()
        guwbeb = cadac_matmul(tblc, guwbel)
        gufspb = cadac_matmul(tblc, gufspl)
        gucbias = float(self.xh[15])
        gucfreq = float(self.xh[16])
        cbias = cbias - gucbias

        store.set("GUSTTCL", gusttcl)
        store.set("GUVBEL", guvbel)
        store.set("GURECEL", gurecel)
        store.set("GUFSPB", gufspb)
        store.set("GUWBEB", guwbeb)
        store.set("gucbias", gucbias)
        store.set("gucfreq", gucfreq)
        store.set("cbias", cbias)
        store.set("mgps", mgps)

        # Fortran: RESET STATE
        self.xh[:] = 0.0

    def terminate(self, vehicle, ctx):
        pass

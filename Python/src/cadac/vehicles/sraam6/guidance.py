from math import atan2, cos, sin, sqrt, tan

import numpy as np

from cadac.constants import AGRAV, DEG
from cadac.kernel.state import Field
from cadac.math.frames import polar_from_cart

SMALL = 1e-7
_ZEROS3 = (0.0, 0.0, 0.0)


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


class Sraam6Guidance:
    name = "guidance"

    def define(self, vehicle):
        store = vehicle.store
        plot = ("plot",)
        scrn = ("scrn",)
        for field in (
            Field("mguid", 0, "int", "data", "guidance"),
            Field("gnav", 0.0, "real", "data", "guidance"),
            Field("ancomx", 0.0, "real", "out", "guidance", scrn),
            Field("alcomx", 0.0, "real", "out", "guidance", scrn),
            Field("gn", 0.0, "real", "diag", "guidance"),
            Field("mnav", 0, "int", "data", "guidance"),
            Field("apny", 0.0, "real", "diag", "guidance"),
            Field("apnz", 0.0, "real", "diag", "guidance"),
            Field("adely", 0.0, "real", "diag", "guidance"),
            Field("adelz", 0.0, "real", "diag", "guidance"),
            Field("all", 0.0, "real", "diag", "guidance", plot),
            Field("ann", 0.0, "real", "diag", "guidance", plot),
            Field("epchta", 0.0, "real", "save", "guidance"),
            Field("WOELC", _ZEROS3, "vec", "out", "guidance"),
            Field("tgoc", 0.0, "real", "diag", "guidance"),
            Field("dtbc", 0.0, "real", "diag", "guidance", plot),
            Field("dvtbc", 0.0, "real", "diag", "guidance", plot),
            Field("psiobcx", 0.0, "real", "diag", "guidance", plot),
            Field("thtobcx", 0.0, "real", "diag", "guidance", plot),
            Field("UTBLC", _ZEROS3, "vec", "out", "guidance"),
            Field("STELC", _ZEROS3, "vec", "diag", "guidance"),
            Field("STBLC", _ZEROS3, "vec", "diag", "guidance"),
            Field("STELM", _ZEROS3, "vec", "save", "guidance"),
            Field("VTELC", _ZEROS3, "vec", "save", "guidance"),
            Field("range", 0.0, "real", "diag", "guidance", plot),
            Field("azimuthx", 0.0, "real", "diag", "guidance", plot),
            Field("elevationx", 0.0, "real", "diag", "guidance", plot),
        ):
            store.define(field)

    def initialize(self, vehicle, ctx):
        pass

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mguid = store.get("mguid")
        mnav = store.get("mnav")
        epchta = store.get("epchta")
        stelm = np.asarray(store.get("STELM"), dtype=float)
        vtelc = np.asarray(store.get("VTELC"), dtype=float)
        time = store.get("time")
        stel = np.asarray(store.get("STEL"), dtype=float)
        vtel = np.asarray(store.get("VTEL"), dtype=float)
        sbel = np.asarray(store.get("SBEL"), dtype=float)

        if mnav == 3:
            mnav = 0
            epchta = time
            stelm = stel.copy()
            vtelc = vtel.copy()
        elif mnav == 0:
            pass
        else:
            raise ValueError(f"unknown mnav {mnav}")

        dtime = time - epchta
        stelc = stelm + vtelc * dtime
        stblc = stelc - sbel

        if mguid == 0:
            pass
        elif mguid == 3:
            self.guidance_mid(vehicle, stblc, vtelc)
        elif mguid == 6:
            self.guidance_term(vehicle)
        else:
            raise ValueError(f"unknown mguid {mguid}")

        stbl = stel - sbel
        polar = polar_from_cart(stbl)
        range_ = float(polar[0])
        azimuthx = float(polar[1]) * DEG
        elevationx = float(polar[2]) * DEG

        store.set("mnav", mnav)
        store.set("epchta", epchta)
        store.set("STELM", stelm)
        store.set("VTELC", vtelc)
        store.set("STELC", stelc)
        store.set("STBLC", stblc)
        store.set("range", range_)
        store.set("azimuthx", azimuthx)
        store.set("elevationx", elevationx)

    def guidance_mid(self, vehicle, stblc, vtelc):
        store = vehicle.store
        gnav = store.get("gnav")
        tbl = np.asarray(store.get("TBL"), dtype=float)
        vbel = np.asarray(store.get("VBEL"), dtype=float)

        dtbc = float(np.sqrt(float(stblc @ stblc)))
        utblc = stblc * (1.0 / dtbc)
        utbbc = tbl @ utblc
        polar = polar_from_cart(utbbc)
        psiobcx = float(polar[1]) * DEG
        thtobcx = float(polar[2]) * DEG
        vtblc = vtelc - vbel
        dvtbc = abs(float(utblc @ vtblc))
        tgoc = dtbc / dvtbc
        woelc = _skew(utblc) @ vtblc * (1.0 / dtbc)
        aapnb = tbl @ (_skew(woelc) @ utblc) * gnav * dvtbc
        ancomx = -float(aapnb[2]) / AGRAV
        alcomx = float(aapnb[1]) / AGRAV

        store.set("ancomx", ancomx)
        store.set("alcomx", alcomx)
        store.set("WOELC", woelc)
        store.set("UTBLC", utblc)
        store.set("tgoc", tgoc)
        store.set("dtbc", dtbc)
        store.set("dvtbc", dvtbc)
        store.set("psiobcx", psiobcx)
        store.set("thtobcx", thtobcx)

    def guidance_term(self, vehicle):
        store = vehicle.store
        gnav = store.get("gnav")
        time = store.get("time")
        stel = np.asarray(store.get("STEL"), dtype=float)
        vtel = np.asarray(store.get("VTEL"), dtype=float)
        thtpb = store.get("thtpb")
        psipb = store.get("psipb")
        sigdpy = store.get("sigdpy")
        sigdpz = store.get("sigdpz")
        tbl = np.asarray(store.get("TBL"), dtype=float)
        fspb = np.asarray(store.get("FSPB"), dtype=float)
        gmax = store.get("gmax")
        trcond = store.get("trcond")
        trcvel = store.get("trcvel")
        sbel = np.asarray(store.get("SBEL"), dtype=float)
        vbel = np.asarray(store.get("VBEL"), dtype=float)

        sbtl = sbel - stel
        dbt = float(np.sqrt(float(sbtl @ sbtl)))
        dum = float(sbtl @ (vbel - vtel))
        dcvel = abs(dum / dbt)

        if time > 3.0:
            if dcvel < trcvel:
                trcond = 1

        fspcb1 = float(fspb[0])
        adely = fspcb1 * tan(psipb) / AGRAV
        adelz = fspcb1 * tan(thtpb) / (cos(psipb) * AGRAV)

        gravb = tbl @ np.array([0.0, 0.0, 1.0])

        gn = gnav * dcvel
        apny = gn * sigdpz / (cos(psipb) * AGRAV)
        apnz = gn * (sigdpz * tan(thtpb) * tan(psipb) + sigdpy / cos(thtpb)) / AGRAV
        all_ = apny + adely - float(gravb[1])
        ann = apnz + adelz + float(gravb[2])

        aa = sqrt(all_ * all_ + ann * ann)
        if aa > gmax:
            aa = gmax
        if abs(ann) < SMALL and abs(all_) < SMALL:
            phi = 0.0
        else:
            phi = atan2(ann, all_)
        alcomx = aa * cos(phi)
        ancomx = aa * sin(phi)

        store.set("ancomx", ancomx)
        store.set("alcomx", alcomx)
        store.set("trcond", trcond)
        store.set("gn", gn)
        store.set("apny", apny)
        store.set("apnz", apnz)
        store.set("adely", adely)
        store.set("adelz", adelz)
        store.set("all", all_)
        store.set("ann", ann)

    def terminate(self, vehicle, ctx):
        pass

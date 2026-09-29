from math import acos, cos, sin, sqrt

import numpy as np

from cadac.constants import DEG, RAD, REARTH
from cadac.kernel.state import Field
from cadac.math.frames import cart_from_pol, mat2tr, polar_from_cart, skew

_ZEROS3 = (0.0, 0.0, 0.0)
_EYE33 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


class Cruise5Seeker:
    name = "seeker"

    def define(self, vehicle):
        store = vehicle.store
        zeros3 = _ZEROS3
        for field in (
            # Task 2 LOS acquire/track
            Field("mseeker", 0, "int", "data/save", "seeker", ("scrn",)),
            Field("acq_range", 0.0, "real", "data", "seeker"),
            Field("range_go", 0.0, "real", "out", "seeker", ("plot", "scrn")),
            Field("STBG", zeros3, "vec", "out", "seeker", ("plot",)),
            Field("WOEB", zeros3, "vec", "out", "seeker"),
            Field("closing_speed", 0.0, "real", "out", "seeker"),
            Field("time_go", 0.0, "real", "out", "seeker", ("plot", "scrn")),
            Field("psisbx", 0.0, "real", "out", "seeker", ("plot", "scrn")),
            Field("thtsbx", 0.0, "real", "out", "seeker", ("plot", "scrn")),
            Field("targ_com_slot", 0, "int", "save", "seeker"),
            Field("UTBB", zeros3, "vec", "out", "seeker"),
            Field("acquisition", 0, "int", "init/save", "seeker", ("scrn",)),
            # Fortran S1 scene-matching (NMAP path)
            Field("nmap", 0, "int", "data", "seeker"),
            Field("nfixm", 0, "int", "data", "seeker"),
            Field("nfix", 0, "int", "init/save", "seeker", ("scrn",)),
            Field("ns11", 0, "int", "init/save", "seeker"),
            Field("ns12", 0, "int", "init/save", "seeker"),
            Field("isetn3", 1, "int", "init/save", "seeker"),
            Field("isetn4", 1, "int", "init/save", "seeker"),
            Field("isetg4", 0, "int", "out", "seeker"),
            Field("way", 0.0, "real", "out", "seeker"),
            Field("racq", 0.0, "real", "data", "seeker"),
            Field("rmin", 0.0, "real", "data", "seeker"),
            Field("SWEL", zeros3, "vec", "data", "seeker"),
            Field("SWRWL", zeros3, "vec", "data", "seeker"),
            Field("SWREL", zeros3, "vec", "out", "seeker"),
            Field("SWALC", zeros3, "vec", "out", "seeker"),
            Field("SWBL", zeros3, "vec", "out", "seeker"),
            Field("randpb", 0.0, "real", "data", "seeker"),
            Field("randtb", 0.0, "real", "data", "seeker"),
            Field("randpc", 0.0, "real", "data", "seeker"),
            Field("randtc", 0.0, "real", "data", "seeker"),
            Field("randdc", 0.0, "real", "data", "seeker"),
            Field("dtimmp", 0.0, "real", "data", "seeker"),
            Field("dtimcr", 0.0, "real", "data", "seeker"),
            Field("dtimfx", 0.0, "real", "diag", "seeker"),
            Field("epchn3", 0.0, "real", "diag", "seeker"),
            Field("epchn4", 0.0, "real", "out", "seeker"),
            Field("fovyaw", 0.0, "real", "data", "seeker"),
            Field("fovpit", 0.0, "real", "data", "seeker"),
            Field("foryaw", 0.0, "real", "data", "seeker"),
            Field("forpit", 0.0, "real", "data", "seeker"),
            Field("dwb", 0.0, "real", "diag", "seeker"),
            Field("psisb", 0.0, "real", "diag", "seeker"),
            Field("thtsb", 0.0, "real", "diag", "seeker"),
            Field("fvyawm", 0.0, "real", "diag", "seeker"),
            Field("fvpitm", 0.0, "real", "diag", "seeker"),
            Field("frpsim", 0.0, "real", "diag", "seeker"),
            Field("frthtm", 0.0, "real", "diag", "seeker"),
            Field("tfvyaw", 0.0, "real", "diag", "seeker"),
            Field("tfvpit", 0.0, "real", "diag", "seeker"),
            Field("tfrpsi", 0.0, "real", "diag", "seeker"),
            Field("tfrtht", 0.0, "real", "diag", "seeker"),
            Field("EWRWS", zeros3, "vec", "diag", "seeker"),
            Field("EWCWRS", zeros3, "vec", "diag", "seeker"),
            Field("EWCWS", zeros3, "vec", "diag", "seeker"),
            Field("EWAS", zeros3, "vec", "diag", "seeker"),
            Field("EWCAS", zeros3, "vec", "diag", "seeker"),
            Field("EWRAS", zeros3, "vec", "diag", "seeker"),
            Field("ewasu", 0.0, "real", "diag", "seeker"),
            # Prior-cycle / epoch save state (Fortran locals retained across S1)
            Field("SBELM", zeros3, "vec", "save", "seeker"),
            Field("TBLM", _EYE33, "mat", "save", "seeker"),
            Field("psiab", 0.0, "real", "save", "seeker"),
            Field("thtab", 0.0, "real", "save", "seeker"),
            Field("dab", 0.0, "real", "save", "seeker"),
            Field("TLCBN", _EYE33, "mat", "save", "seeker"),
            Field("SWBLCN", zeros3, "vec", "save", "seeker"),
        ):
            if field.name not in store:
                store.define(field)

    def initialize(self, vehicle, ctx):
        # Fortran S1I
        store = vehicle.store
        store.set("ns11", 0)
        store.set("ns12", 0)
        store.set("nfix", 0)
        store.set("isetn3", 1)
        store.set("isetn4", 1)
        store.set("way", 0.0)
        if "SBEL" in store:
            store.set("SBELM", np.asarray(store.get("SBEL"), dtype=float).copy())
        if "TBL" in store:
            store.set("TBLM", np.asarray(store.get("TBL"), dtype=float).copy())

    def terminate(self, vehicle, ctx):
        pass

    def seeker_grnd_ranges(self, vehicle, combus):
        store = vehicle.store
        lon_c = store.get("lonx") * RAD
        lat_c = store.get("latx") * RAD
        slots = []
        ranges = []
        for i, packet in enumerate(combus):
            if packet.type != "TARGET3":
                continue
            lon_t = packet.vars["lonx"] * RAD
            lat_t = packet.vars["latx"] * RAD
            dum = sin(lat_t) * sin(lat_c) + cos(lat_t) * cos(lat_c) * cos(
                lon_t - lon_c
            )
            slots.append(i)
            ranges.append(REARTH * acos(dum))
        return slots, ranges

    def execute(self, vehicle, ctx):
        store = vehicle.store
        mseeker = store.get("mseeker")
        if mseeker == 0:
            if "SBEL" in store and "TBL" in store:
                store.set("SBELM", np.asarray(store.get("SBEL"), dtype=float).copy())
                store.set("TBLM", np.asarray(store.get("TBL"), dtype=float).copy())
            return

        nmap = store.get("nmap")
        if nmap != 0:
            self._execute_scene(vehicle, ctx)
            return

        if mseeker not in (1, 3):
            raise ValueError(f"unknown mseeker {mseeker}")

        acq_range = store.get("acq_range")
        acquisition = store.get("acquisition")
        targ_com_slot = store.get("targ_com_slot")
        combus = ctx.combus

        if not acquisition and mseeker == 1:
            slots, ranges = self.seeker_grnd_ranges(vehicle, combus)
            for slot, range_m in zip(slots, ranges):
                if range_m < acq_range:
                    acquisition = 1
                    mseeker = 3
                    targ_com_slot = slot

        if mseeker == 3:
            packet = combus[targ_com_slot]
            tgt_vbeg = packet.vars["vbeg"]
            tgt_sbii = packet.vars["sbii"]
            tig = store.get("tig")
            vbeg = store.get("vbeg")
            sbii = store.get("sbii")
            tbg = store.get("TBG")

            stbi = tgt_sbii - sbii
            tgi = tig.T
            stbg = tgi @ stbi
            range_go = sqrt(float(stbg[0] ** 2 + stbg[1] ** 2 + stbg[2] ** 2))
            inv_dtb = 1.0 / range_go
            utbg = stbg * inv_dtb
            vtbg = tgt_vbeg - vbeg
            woeb = tbg @ skew(utbg) @ vtbg * inv_dtb
            vbtg = vtbg * (-1.0)
            closing_speed = float(utbg @ vbtg)
            time_go = range_go / closing_speed
            utbb = tbg @ utbg
            polar = polar_from_cart(utbb)
            psisbx = polar[1] * DEG
            thtsbx = polar[2] * DEG

            store.set("range_go", range_go)
            store.set("STBG", stbg)
            store.set("WOEB", woeb)
            store.set("closing_speed", closing_speed)
            store.set("time_go", time_go)
            store.set("psisbx", psisbx)
            store.set("thtsbx", thtsbx)
            store.set("UTBB", utbb)

        store.set("targ_com_slot", targ_com_slot)
        store.set("acquisition", acquisition)
        store.set("mseeker", mseeker)

    def _s1epch(self, store, t, nfix):
        """Fortran S1EPCH — measurement and update epochs."""
        dtimmp = store.get("dtimmp")
        dtimcr = store.get("dtimcr")
        if nfix == 1:
            epchn3 = dtimmp + t
            dtimfx = dtimmp + dtimcr
            epchn4 = dtimfx + t
        else:
            epchn3 = t
            dtimfx = dtimcr
            epchn4 = dtimfx + t
        store.set("epchn3", epchn3)
        store.set("epchn4", epchn4)
        store.set("dtimfx", dtimfx)
        return epchn3, epchn4, dtimfx

    def _s1los(
        self,
        store,
        t,
        swbl,
        dwb,
        swblc,
        swrel,
        sbelm,
        tblm,
        nfix,
        nfixm,
        mseeker,
        rmin,
        fvyawm,
        fvpitm,
        frpsim,
        frthtm,
    ):
        """Fortran S1LOS — boresight, FOV/FOR diagnostics, plane errors."""
        tblc = np.asarray(store.get("TBLC"), dtype=float)
        randpb = store.get("randpb")
        randtb = store.get("randtb")
        fovyaw = store.get("fovyaw")
        fovpit = store.get("fovpit")
        foryaw = store.get("foryaw")
        forpit = store.get("forpit")

        swcbb = tblc @ swblc
        polar = polar_from_cart(swcbb)
        dwbc = float(polar[0])
        psiwb = float(polar[1])
        thtwb = float(polar[2])

        psisb = psiwb + randpb
        thtsb = thtwb + randtb

        uabs = np.array([1.0, 0.0, 0.0])
        tsb = mat2tr(psisb, thtsb)
        tsl = tsb @ tblm
        swbs = tsl @ swbl
        uwbs = swbs * (1.0 / dwb)
        swcbs = tsb @ swcbb
        uwcbs = swcbs * (1.0 / dwbc)
        swrbl = swrel - sbelm
        swrbs = tsl @ swrbl
        dwrb = float(np.linalg.norm(swrbs))
        uwrbs = swrbs * (1.0 / dwrb)

        dumwas = uwbs - uabs

        ewas = np.asarray(store.get("EWAS"), dtype=float).copy()
        ewasu = store.get("ewasu")
        ewrws = np.asarray(store.get("EWRWS"), dtype=float).copy()
        ewcwrs = np.asarray(store.get("EWCWRS"), dtype=float).copy()
        ewcws = np.asarray(store.get("EWCWS"), dtype=float).copy()
        ewcas = np.asarray(store.get("EWCAS"), dtype=float).copy()
        ewras = np.asarray(store.get("EWRAS"), dtype=float).copy()

        if abs(dumwas[1]) < fovyaw and abs(dumwas[2]) < fovpit:
            ewas = dumwas.copy()
            ewasu = -ewas[2]
            ewrws = uwrbs - uwbs
            ewcwrs = uwcbs - uwrbs
            ewcws = uwcbs - uwbs
            ewcas = uwcbs - uabs
            ewras = uwrbs - uabs

        # Fortran: (NFIX.GT.NFIXM.AND.MSEEK.EQ.1).OR.(DWB.LT.RMIN)
        if (nfix > nfixm and mseeker == 1) or dwb < rmin:
            ewas = np.zeros(3)
            ewasu = 0.0
            ewrws = np.zeros(3)
            ewcwrs = np.zeros(3)
            ewcws = np.zeros(3)
            ewcas = np.zeros(3)
            ewras = np.zeros(3)
            psisb = 0.0
            thtsb = 0.0

        deaws2 = abs(ewas[1])
        if deaws2 > fvyawm:
            fvyawm = deaws2
            store.set("tfvyaw", t)
        deaws3 = abs(ewas[2])
        if deaws3 > fvpitm:
            fvpitm = deaws3
            store.set("tfvpit", t)
        dpsisb = abs(psisb)
        if dpsisb > frpsim:
            frpsim = dpsisb
            store.set("tfrpsi", t)
        dthtsb = abs(thtsb)
        if dthtsb > frthtm:
            frthtm = dthtsb
            store.set("tfrtht", t)

        if abs(dumwas[1]) > fovyaw or abs(dumwas[2]) > fovpit:
            nfix = 777
        if abs(psisb) > foryaw or abs(thtsb) > forpit:
            nfix = 888

        trfory = store.get_optional("trfory", None)
        if trfory is not None:
            trforp = store.get("trforp")
            trfovy = store.get("trfovy")
            trfovp = store.get("trfovp")
            if abs(dumwas[1]) > trfovy:
                store.set("trcode", 4.0)
            if abs(dumwas[2]) > trfovp:
                store.set("trcode", 5.0)
            if abs(psisb) > trfory:
                store.set("trcode", 6.0)
            if abs(thtsb) > trforp:
                store.set("trcode", 7.0)

        store.set("EWAS", ewas)
        store.set("ewasu", ewasu)
        store.set("EWRWS", ewrws)
        store.set("EWCWRS", ewcwrs)
        store.set("EWCWS", ewcws)
        store.set("EWCAS", ewcas)
        store.set("EWRAS", ewras)

        return (
            fvyawm,
            fvpitm,
            frpsim,
            frthtm,
            dwbc,
            psisb,
            thtsb,
            nfix,
        )

    def _execute_scene(self, vehicle, ctx):
        """Fortran S1 imaging scene-match sequence (MSEEK 1→2→3→4)."""
        store = vehicle.store
        mseeker = store.get("mseeker")
        t = float(ctx.sim_time)

        sbel = np.asarray(store.get("SBEL"), dtype=float)
        tbl = np.asarray(store.get("TBL"), dtype=float)
        sbelm = np.asarray(store.get("SBELM"), dtype=float).copy()
        tblm = np.asarray(store.get("TBLM"), dtype=float).copy()

        swel = np.asarray(store.get("SWEL"), dtype=float)
        swrwl = np.asarray(store.get("SWRWL"), dtype=float)
        sbwlc = np.asarray(store.get("SBWLC"), dtype=float)
        tblc = np.asarray(store.get("TBLC"), dtype=float)

        swbl = swel - sbelm
        dwb = float(np.linalg.norm(swbl))
        swrel = swrwl + swel
        swblc = (-1.0) * sbwlc

        store.set("way", 0.0)
        store.set("SWBL", swbl)
        store.set("dwb", dwb)
        store.set("SWREL", swrel)

        nmap = store.get("nmap")
        ns11 = store.get("ns11")
        ns12 = store.get("ns12")
        nfix = store.get("nfix")
        nfixm = store.get("nfixm")
        racq = store.get("racq")
        rmin = store.get("rmin")
        isetn3 = store.get("isetn3")
        isetn4 = store.get("isetn4")
        fvyawm = store.get("fvyawm")
        fvpitm = store.get("fvpitm")
        frpsim = store.get("frpsim")
        frthtm = store.get("frthtm")
        psisb = store.get("psisb")
        thtsb = store.get("thtsb")
        epchn3 = store.get("epchn3")
        epchn4 = store.get("epchn4")

        if dwb < racq:
            if nmap != ns11:
                ns11 = nmap
                nfix = 1
                store.set("isetg4", 1)
                fvyawm = 0.0
                fvpitm = 0.0
                frpsim = 0.0
                frthtm = 0.0

            (
                fvyawm,
                fvpitm,
                frpsim,
                frthtm,
                dwbc,
                psisb,
                thtsb,
                nfix,
            ) = self._s1los(
                store,
                t,
                swbl,
                dwb,
                swblc,
                swrel,
                sbelm,
                tblm,
                nfix,
                nfixm,
                mseeker,
                rmin,
                fvyawm,
                fvpitm,
                frpsim,
                frthtm,
            )

            if nfix != ns12 and nfix <= nfixm and dwb > rmin:
                ns12 = nfix
                mseeker = 2
                isetn3 = 0
                isetn4 = 0
                epchn3, epchn4, _ = self._s1epch(store, t, nfix)

            if t >= epchn3 and isetn3 == 0:
                isetn3 = 1
                mseeker = 3
                store.set("psiab", psisb)
                store.set("thtab", thtsb)
                store.set("dab", dwbc)
                tlcb = tblc.T.copy()
                tlcbn = tlcb.copy()
                store.set("TLCBN", tlcbn)
                tlcln = tlcbn @ tblm
                swblcn = tlcln @ swbl
                store.set("SWBLCN", swblcn)

            if t >= epchn4 and isetn4 == 0:
                isetn4 = 1
                mseeker = 4
                ns12 = 0
                nfix = nfix + 1
                psiabk = store.get("psiab") + store.get("randpc")
                thtabk = store.get("thtab") + store.get("randtc")
                dabk = store.get("dab") + store.get("randdc")
                sabb = cart_from_pol(dabk, psiabk, thtabk)
                tlcbn = np.asarray(store.get("TLCBN"), dtype=float)
                sablc = tlcbn @ sabb
                swblcn = np.asarray(store.get("SWBLCN"), dtype=float)
                swalc = swblcn - sablc
                store.set("SWALC", swalc)

        store.set("SBELM", sbel.copy())
        store.set("TBLM", tbl.copy())
        store.set("psisb", psisb)
        store.set("thtsb", thtsb)
        store.set("psisbx", psisb * DEG)
        store.set("thtsbx", thtsb * DEG)
        store.set("fvyawm", fvyawm)
        store.set("fvpitm", fvpitm)
        store.set("frpsim", frpsim)
        store.set("frthtm", frthtm)
        store.set("ns11", ns11)
        store.set("ns12", ns12)
        store.set("nfix", nfix)
        store.set("isetn3", isetn3)
        store.set("isetn4", isetn4)
        store.set("mseeker", mseeker)
        store.set("epchn3", epchn3)
        store.set("epchn4", epchn4)

"""CRUISE5 environmental module G2 — MAIR=|MATM|MWIND| packing."""

from cadac.eom.round3 import Round3Environment
from cadac.kernel.state import Field


class Cruise5Environment(Round3Environment):
    """CRUISE5 G2: MATM=INT(MAIR/10), MWIND=MAIR-MATM*10 (not GHAME3 MAIR=1)."""

    fields = Round3Environment.fields + (
        Field("dvael", 0.0, "real", "data", "environment"),
        Field("psiwlx", 0.0, "real", "data", "environment"),
        Field("dvae3", 0.0, "real", "data", "environment"),
        Field("VAEL", (0.0, 0.0, 0.0), "vec", "state", "environment"),
        Field("VAELD", (0.0, 0.0, 0.0), "vec", "state", "environment"),
        Field("dvba", 0.0, "real", "out", "environment"),
        Field("dvw", 0.0, "real", "diag", "environment"),
    )

    def __init__(self, weather_deck=None) -> None:
        super().__init__(weather_deck=weather_deck, mair_pack="cruise5")

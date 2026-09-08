from cadac.constants import EARTH_MASS, REARTH, G


def gravity(alt_m: float) -> float:
    return G * EARTH_MASS / (REARTH + alt_m) ** 2

from cadac.constants import G, EARTH_MASS, REARTH


def gravity(alt_m):
    return G * EARTH_MASS / (REARTH + alt_m) ** 2

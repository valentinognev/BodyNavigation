"""CADAC++ unituni / gauss / Dryden white noise via glibc rand()."""

import ctypes
import math
from dataclasses import dataclass

from cadac.constants import PI

# CADAC execution.cpp: int iseed=0; srand(iseed) when nmc==0. AGM6 test-case
# ASC is MONTE 1 12345 (harvest keeps iseed). Dryden always calls rand().
DEFAULT_ISEED = 0
# Missile::def_ins 18 gauss() + init_ins 9 gauss(0,1) before first environment.
_INS_GAUSS_DRAWS = 27
# C++ def_ins sigma order (3-vectors).
INS_DEFINE_SIGMAS = (
    (1.1e-4, 1.1e-4, 1.1e-4),
    (2e-5, 2.5e-5, 2.5e-5),
    (1e-5, 3.2e-6, 3.2e-6),
    (1.1e-4, 1.1e-4, 1.1e-4),
    (5e-4, 5e-4, 5e-4),
    (3.56e-3, 3.56e-3, 3.56e-3),
)
# AGM6 MARKOV cards (randal, randt, randp, randeh). markov_noise calls gauss()
# even when nmonte==0, then zeros the stored values.
_AGM6_MARKOV_COUNT = 4
AGM6_MARKOV_NAMES = ("randal", "randt", "randp", "randeh")
# ROCKET6 insertion: 9 GPS + 6 startrack MARKOV cards. Same nmonte==0 draw-then-zero.
ROCKET6_MARKOV_COUNT = 15

_libc = ctypes.CDLL("libc.so.6")
_libc.srand.argtypes = [ctypes.c_uint]
_libc.rand.restype = ctypes.c_int
RAND_MAX = 2147483647

_iset = 0
_gset = 0.0
_seeded = False
_ins_burned = False
# Set by markov_noise; prepare_for_dryden skips stand-in burn+zero when True.
_markov_noise_drew = False


@dataclass
class MarkovEntry:
    """One C++ Markov list slot (sigma/bcor from MARKOV card; saved across steps)."""

    name: str
    sigma: float
    bcor: float
    saved: float = 0.0
    status: bool = True


def seed(iseed=DEFAULT_ISEED):
    global _iset, _gset, _seeded, _ins_burned, _markov_noise_drew
    _libc.srand(int(iseed))
    _iset = 0
    _gset = 0.0
    _seeded = True
    _ins_burned = False
    _markov_noise_drew = False


def unituni():
    return _libc.rand() / RAND_MAX


def uniform(min_v, max_v):
    return min_v + (max_v - min_v) * unituni()


def gauss(mean, sig):
    global _iset, _gset
    if _iset == 0:
        while True:
            v1 = 2.0 * unituni() - 1.0
            v2 = 2.0 * unituni() - 1.0
            rsq = v1 * v1 + v2 * v2
            if rsq < 1.0 and rsq != 0.0:
                break
        fac = math.sqrt(-2.0 * math.log(rsq) / rsq)
        _gset = v1 * fac
        _iset = 1
        value = v2 * fac
    else:
        _iset = 0
        value = _gset
    return value * sig + mean


def dryden_white(int_step):
    while True:
        value1 = unituni()
        if value1 != 0.0:
            break
    value2 = unituni()
    return (1.0 / math.sqrt(int_step)) * math.sqrt(2.0 * math.log(1.0 / value1)) * math.cos(
        2.0 * PI * value2
    )


def markov(sigma, bcor, time, int_step, value_saved):
    """CADAC utility markov(); returns (value, updated value_saved)."""
    value = gauss(0.0, sigma)
    if time == 0.0:
        value_saved = value
    elif bcor != 0.0:
        dum = math.exp(-bcor * int_step)
        dumsqrd = dum * dum
        value = value * math.sqrt(1.0 - dumsqrd) + value_saved * dum
        value_saved = value
    return value, value_saved


def markov_noise(store, markov_list, time, int_step, nmonte):
    """Refresh MARKOV deck variables (C++ vehicle::markov_noise before modules).

    Always draws via markov(); when nmonte==0, store is forced to 0 after the draw
    (saved still advances). Matches AGM6/HYPER6/ROCKET6/SRAAM6/SAM6 execution.cpp.
    """
    global _markov_noise_drew
    if not markov_list:
        return
    for entry in markov_list:
        if not getattr(entry, "status", True):
            continue
        value, saved = markov(
            entry.sigma, entry.bcor, time, int_step, entry.saved
        )
        store.set(entry.name, value)
        # C++ set_markov_saved(module_variable.real()) after gets(markov(...))
        entry.saved = float(store.get(entry.name))
        if not nmonte:
            store.set(entry.name, 0.0)
    _markov_noise_drew = True


def clear_markov_noise_drew():
    """Clear stand-in skip flag after a vehicle step (no Dryden path)."""
    global _markov_noise_drew
    _markov_noise_drew = False


def mark_ins_stream_consumed():
    global _ins_burned
    _ins_burned = True


def draw_ins_define_errors():
    # g++ evaluates Variable::init(name, v1, v2, v3, ...) args right-to-left.
    vectors = []
    for sigs in INS_DEFINE_SIGMAS:
        third = gauss(0.0, sigs[2])
        second = gauss(0.0, sigs[1])
        first = gauss(0.0, sigs[0])
        vectors.append((first, second, third))
    return tuple(vectors)


def draw_ins_init_unit():
    draws = tuple(gauss(0.0, 1.0) for _ in range(9))
    mark_ins_stream_consumed()
    return draws


def prepare_for_dryden(store=None, markov_count=_AGM6_MARKOV_COUNT):
    global _ins_burned, _markov_noise_drew
    if not _seeded:
        seed(DEFAULT_ISEED)
    if not _ins_burned:
        for _ in range(_INS_GAUSS_DRAWS):
            gauss(0.0, 1.0)
        _ins_burned = True
    if _markov_noise_drew:
        # Live markov_noise already advanced gauss and set/zeroed MARKOV stores.
        _markov_noise_drew = False
        return
    for _ in range(markov_count):
        gauss(0.0, 1.0)
    if store is not None:
        for name in AGM6_MARKOV_NAMES:
            if name in store:
                store.set(name, 0.0)

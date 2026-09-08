"""CADAC++ unituni / gauss / Dryden white noise via glibc rand()."""

import ctypes
import math

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


def seed(iseed=DEFAULT_ISEED):
    global _iset, _gset, _seeded, _ins_burned
    _libc.srand(int(iseed))
    _iset = 0
    _gset = 0.0
    _seeded = True
    _ins_burned = False


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
    global _ins_burned
    if not _seeded:
        seed(DEFAULT_ISEED)
    if not _ins_burned:
        for _ in range(_INS_GAUSS_DRAWS):
            gauss(0.0, 1.0)
        _ins_burned = True
    for _ in range(markov_count):
        gauss(0.0, 1.0)
    if store is not None:
        names = store.names()
        for name in AGM6_MARKOV_NAMES:
            if name in names:
                store.set(name, 0.0)

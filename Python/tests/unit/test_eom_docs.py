import inspect

import cadac.eom.flat0 as flat0
import cadac.eom.flat3 as flat3
import cadac.eom.flat6 as flat6
import cadac.eom.round3 as round3
import cadac.eom.round6 as round6
import cadac.eom.rotor as rotor

_CLASSES = [
    flat6.Flat6Environment,
    flat6.Flat6Kinematics,
    flat6.Flat6Euler,
    flat6.Flat6Newton,
    round6.Round6Environment,
    round6.Round6Kinematics,
    round6.Round6Euler,
    round6.Round6Newton,
    round3.Round3Environment,
    round3.Round3Newton,
    flat3.Flat3Environment,
    flat3.Flat3Kinematics,
    flat3.Flat3Newton,
    flat0.Flat0Kinematics,
    flat0.Flat0Newton,
    rotor.RotorEnvironment,
    rotor.RotorTrajectory,
    rotor.RotorAttitude,
]


def test_eom_classes_have_docstrings():
    missing = [cls.__name__ for cls in _CLASSES if not cls.__doc__]
    assert missing == []


def test_eom_class_docs_mention_cadac_or_zipfel():
    weak = [
        cls.__name__
        for cls in _CLASSES
        if not cls.__doc__
        or ("cadac" not in cls.__doc__.lower() and "zipfel" not in cls.__doc__.lower())
    ]
    assert weak == []


def test_round6_euler_comment_points_at_cadac_inverse():
    src = inspect.getsource(round6.Round6Euler.execute)
    assert "cadac_inverse" in src
    assert "np.linalg.inv" in src
    assert "np.linalg.inv(" not in src

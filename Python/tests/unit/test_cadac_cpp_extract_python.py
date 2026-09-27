from cadac_cpp.extract_python import (
    extract_family_keys,
    extract_global_types,
    extract_python_modes,
)

CLI_SNIPPET = '''
_VEHICLE_TYPES = {
    "CRUISE3": Cruise3,
    "HYPER6": Hyper6,
    "TARGET3": Target3,
}
_VEHICLE_FAMILIES: dict[tuple[str, str], type] = {
    ("aim5", "AIM5"): Aim5,
    ("rocket6", "HYPER6"): Rocket6,
}
'''

CONTROL_SNIPPET = '''
        if maut == 0:
            return
        if maut not in (24,):
            raise ValueError(f"unknown maut {maut}")
        if mauty == 2:
            pass
'''


def test_global_types():
    assert extract_global_types(CLI_SNIPPET) == ["CRUISE3", "HYPER6", "TARGET3"]


def test_family_keys():
    assert extract_family_keys(CLI_SNIPPET) == [("aim5", "AIM5"), ("rocket6", "HYPER6")]


REGISTER_SNIPPET = '''
register_family_type("sam6", "MISSILE6", Sam6Missile)
register_family_type("sam6", "AIRCRAFT3", Sam6Aircraft)
register_family_type("sam6", "ROCKET5", Sam6Rocket)
register_family_type("sam6", "RADAR0", Sam6Radar)
register_family_type("sraam6", "MISSILE6", Sraam6Missile)
register_family_type("sraam6", "TARGET3", Sraam6Target)
register_family_type("agm6", "MISSILE6", Agm6Missile)
register_family_type("agm6", "TARGET3", Agm6Target)
register_family_type("agm6", "AIRCRAFT3", Agm6Aircraft)
'''


def test_family_keys_from_register_family_type():
    keys = extract_family_keys(CLI_SNIPPET + REGISTER_SNIPPET)
    assert ("aim5", "AIM5") in keys
    assert ("rocket6", "HYPER6") in keys
    assert ("sam6", "MISSILE6") in keys
    assert ("sam6", "AIRCRAFT3") in keys
    assert ("sam6", "ROCKET5") in keys
    assert ("sam6", "RADAR0") in keys
    assert ("sraam6", "MISSILE6") in keys
    assert ("sraam6", "TARGET3") in keys
    assert ("agm6", "MISSILE6") in keys
    assert ("agm6", "TARGET3") in keys
    assert ("agm6", "AIRCRAFT3") in keys


def test_python_modes_implemented_and_stubbed():
    implemented, stub_flags = extract_python_modes(CONTROL_SNIPPET)
    assert ("maut", 0) in implemented
    assert ("maut", 24) in implemented
    assert ("mauty", 2) in implemented
    assert "maut" in stub_flags


def test_python_modes_not_equal_guard_counts_as_implemented():
    text = """
        if mguide != 5:
            raise ValueError(f"unknown mguide {mguide}")
        if mterm != -1:
            pass
        if mins not in (0, 1):
            raise ValueError(f"unknown mins {mins}")
    """
    implemented, _stub_flags = extract_python_modes(text)
    assert ("mguide", 5) in implemented
    assert ("mterm", -1) in implemented
    assert ("mins", 0) in implemented
    assert ("mins", 1) in implemented

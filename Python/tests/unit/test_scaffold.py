# Python/tests/unit/test_scaffold.py
import cadac

def test_package_importable():
    assert cadac.__name__ == "cadac"

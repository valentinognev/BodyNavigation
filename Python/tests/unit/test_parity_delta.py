import json
from pathlib import Path

# Ruling R4: pytest cwd is Python/, so do not use a repo-root path
# "Python/tools/...". Resolve from this file (Python/tests/unit/).
_INVENTORY = Path(__file__).resolve().parents[2] / "tools" / "cadac_cpp" / "inventory.json"


def test_no_missing_vehicles():
    rows = json.load(open(_INVENTORY))
    missing = [r for r in rows if r["kind"] == "vehicle" and r["status"] == "missing"]
    assert missing == []

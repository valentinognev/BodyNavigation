"""SAM6 7-vehicle RF e2e is optional.

Skip if tests/e2e/goldens/sam6/rf/plot.csv is absent. Do not require that
golden; do not fail CI for its absence.
"""

from pathlib import Path

import pytest

GOLDEN = Path(__file__).resolve().parent / "goldens" / "sam6" / "rf" / "plot.csv"


def require_golden(path: Path):
    if not path.is_file():
        pytest.skip("SAM6 RF golden plot.csv is absent")


def test_require_golden_skips_when_missing(tmp_path):
    with pytest.raises(pytest.skip.Exception, match="SAM6 RF golden plot.csv is absent"):
        require_golden(tmp_path / "plot.csv")


def test_require_golden_continues_when_present(tmp_path):
    path = tmp_path / "plot.csv"
    path.write_text("x\n", encoding="utf-8", newline="\n")
    require_golden(path)


def test_rf_e2e_skips_without_golden():
    require_golden(GOLDEN)

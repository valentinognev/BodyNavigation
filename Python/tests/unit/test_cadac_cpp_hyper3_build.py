import os
import shutil
from pathlib import Path

import pytest

from cadac_cpp.build_cadac import build_program


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ missing")
def test_hyper3_binary_exists():
    binary = build_program("HYPER3")
    assert binary.is_file()
    assert os.access(binary, os.X_OK)

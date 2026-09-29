"""comscrn.asc flatten must expand length-3 packet vectors (incl. lowercase sbii)."""

from io import StringIO

import numpy as np

from cadac.io.comscrn import _flatten_values, write_comscrn_data
from cadac.kernel.combus import Packet


def test_comscrn_flatten_expands_lowercase_sbii():
    packet = Packet(
        name="c1",
        type="CRUISE3",
        status=1,
        vars={
            "time": 0.0,
            "sbii": np.array([1.0, 2.0, 3.0]),
            "alt": 1000.0,
        },
    )
    assert _flatten_values(packet) == [1.0, 2.0, 3.0, 1000.0]


def test_write_comscrn_data_emits_sbii_components_without_typeerror():
    packet = Packet(
        name="c1",
        type="CRUISE3",
        status=1,
        vars={
            "time": 0.0,
            "sbii": np.array([1.0, 2.0, 3.0]),
            "alt": 1000.0,
        },
    )
    stream = StringIO()
    write_comscrn_data(stream, [packet], 0.0)
    text = stream.getvalue()
    assert "sbii1" in text and "sbii2" in text and "sbii3" in text
    assert "1.0" in text and "2.0" in text and "3.0" in text

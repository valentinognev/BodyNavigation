"""traj.asc flatten must expand length-3 packet vectors (incl. lowercase sbii)."""

from io import StringIO

import numpy as np

from cadac.io.traj import (
    _flatten_packet_values,
    nvariables,
    write_traj_banner,
    write_traj_data,
)
from cadac.kernel.combus import Packet


def _packet_sbii():
    return Packet(
        name="c1",
        type="CRUISE3",
        status=1,
        vars={
            "time": 0.0,
            "sbii": np.array([1.0, 2.0, 3.0]),
            "alt": 1000.0,
        },
    )


def test_flatten_expands_lowercase_sbii_to_three_floats():
    values = _flatten_packet_values(_packet_sbii(), skip_time=True)
    assert values == [1.0, 2.0, 3.0, 1000.0]


def test_flatten_still_expands_uppercase_SBII():
    packet = Packet(
        name="m1",
        type="HYPER6",
        status=1,
        vars={"time": 0.0, "SBII": np.array([4.0, 5.0, 6.0])},
    )
    assert _flatten_packet_values(packet, skip_time=True) == [4.0, 5.0, 6.0]


def test_nvariables_and_banner_count_sbii_as_three_columns():
    combus = [_packet_sbii()]
    # names: time, sbii(+2), alt → 1 + 3 + 1 = 5 after shared-time rule on single packet
    assert nvariables(combus) == 5
    stream = StringIO()
    write_traj_banner(stream, "t", combus)
    text = stream.getvalue()
    assert "sbii1_" in text and "sbii2_" in text and "sbii3_" in text


def test_write_traj_data_emits_sbii_components_without_typeerror():
    stream = StringIO()
    write_traj_data(stream, [_packet_sbii()], merge=False)
    text = stream.getvalue()
    assert "1.0" in text and "2.0" in text and "3.0" in text
    assert "1000" in text

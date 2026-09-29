"""CADAC comscrn.asc writers (comscrn_data from combus → text dump)."""

from __future__ import annotations

from pathlib import Path

from cadac.kernel.combus import Packet

_LABEL_LEN = 14
_FIELD_WIDTH = 15


def _truncate_label(name: str) -> str:
    return name[:_LABEL_LEN]


def _is_vector_name(name: str) -> bool:
    return bool(name) and name[0].isupper()


def _is_vector_entry(name: str, value) -> bool:
    if _is_vector_name(name):
        return True
    if isinstance(value, (str, bytes, bytearray)):
        return False
    try:
        return len(value) == 3
    except TypeError:
        return False


def _packet_names(packet: Packet) -> list[str]:
    return list(packet.vars.keys())


def _active_packets(combus: list[Packet]) -> list[Packet]:
    return [p for p in combus if p.vars]


def _group_by_type(combus: list[Packet]) -> list[tuple[str, list[Packet]]]:
    """Preserve first-seen type order; skip empty packets."""
    groups: list[tuple[str, list[Packet]]] = []
    index: dict[str, int] = {}
    for packet in _active_packets(combus):
        vtype = packet.type or "UNKNOWN"
        if vtype not in index:
            index[vtype] = len(groups)
            groups.append((vtype, []))
        groups[index[vtype]][1].append(packet)
    return groups


def _row_prefix(packet: Packet) -> str:
    """C++ uses leading char of vehicle id (m_/r_/a_/…)."""
    name = packet.name or "?"
    return name[0]


def _flatten_values(packet: Packet) -> list[float]:
    """Values excluding 'time'; ints→float, vectors→3 components."""
    values: list[float] = []
    for name, value in packet.vars.items():
        if name == "time":
            continue
        if _is_vector_entry(name, value):
            values.extend(float(value[i]) for i in range(3))
        elif isinstance(value, int) and not isinstance(value, bool):
            values.append(float(value))
        else:
            values.append(float(value))
    return values


def write_comscrn_data(
    stream,
    combus: list[Packet],
    sim_time: float,
) -> None:
    """Append one CADAC comscrn_data dump (time banner + type sections)."""
    # C++: cout<<" time = ";cout.width(6);cout<<sim_time; ... combus ...
    stream.write(f" time = {sim_time:<6}")
    stream.write(
        " ------------------------------------ combus ----------------------------------------------------------"
    )
    for vtype, packets in _group_by_type(combus):
        # C++: cout.width(16); cout<<"\n** TYPE **"; then labels (eight across)
        stream.write(f"\n{'** ' + vtype + ' **':<16}")
        first = packets[0]
        m = 1
        for name in _packet_names(first):
            if name == "time":
                continue
            buff = _truncate_label(name)
            if _is_vector_entry(name, first.vars[name]):
                for n in range(1, 4):
                    pad = max(1, _FIELD_WIDTH - len(buff))
                    stream.write(f"{buff}{n:<{pad}}")
                    m += 1
                    if m > 7:
                        m = 0
                        stream.write("\n")
            else:
                stream.write(f"{buff:<{_FIELD_WIDTH}}")
                m += 1
                if m > 7:
                    m = 0
                    stream.write("\n")
        stream.write("\n")
        # Per-vehicle data rows
        for idx, packet in enumerate(packets, start=1):
            prefix = _row_prefix(packet)
            stream.write(f"\n *** {prefix}_{idx:<8}")
            k = 1
            for value in _flatten_values(packet):
                if k > 7:
                    k = 0
                    stream.write("\n")
                stream.write(f"{value:<{_FIELD_WIDTH}}")
                k += 1
    stream.write("\n")


def open_comscrn_stream(out_dir: Path):
    """Open comscrn.asc for writing (Python text artifact for C++ cout dump)."""
    path = Path(out_dir) / "comscrn.asc"
    stream = path.open("w", encoding="utf-8", newline="\n")
    return path, stream

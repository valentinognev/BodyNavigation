"""CADAC traj.asc writers (traj_banner + traj_data from combus)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from cadac.kernel.combus import Packet, packet_from_store

_LABEL_LEN = 10
_FIELD_WIDTH = 16
_ACROSS = 5


def _truncate_label(name: str) -> str:
    return name[:_LABEL_LEN]


def _is_vector_name(name: str) -> bool:
    return bool(name) and name[0].isupper()


def _packet_names(packet: Packet) -> list[str]:
    return list(packet.vars.keys())


def _active_packets(combus: list[Packet]) -> list[Packet]:
    return [p for p in combus if p.vars]


def nvariables(combus: list[Packet]) -> int:
    """CADAC nvariables: expand vectors, keep a single shared 'time' label."""
    packets = _active_packets(combus)
    if not packets:
        return 0
    total = 0
    for packet in packets:
        names = _packet_names(packet)
        nvec = sum(1 for name in names if _is_vector_name(name))
        total += len(names) + 2 * nvec
    return total - (len(packets) - 1)


def packets_from_vehicles(vehicles: list) -> list[Packet]:
    """Build combus-shaped packets from vehicle stores (post-init snapshot)."""
    packets: list[Packet] = []
    for vehicle in vehicles:
        com_names = getattr(vehicle, "com_names", None) or []
        store = getattr(vehicle, "store", None)
        if store is None or not com_names:
            packets.append(
                Packet(
                    name=getattr(vehicle, "name", ""),
                    type=getattr(vehicle, "type", ""),
                    status=getattr(vehicle, "health", 1),
                    vars={},
                )
            )
            continue
        packet = packet_from_store(store, com_names)
        packet.name = getattr(vehicle, "name", "")
        packet.type = getattr(vehicle, "type", "")
        packet.status = getattr(vehicle, "health", 1)
        packets.append(packet)
    return packets


def write_traj_banner(
    stream,
    title: str,
    combus: list[Packet],
    *,
    when: datetime | None = None,
) -> int:
    """Write CADAC traj_banner header. Returns nvariables."""
    when = when or datetime.now()
    date_s = f"{when.strftime('%b')} {when.day:2d} {when.year}"
    time_s = when.strftime("%H:%M:%S")
    nvar = nvariables(combus)
    stream.write(f"1{title} {date_s} {time_s}\n")
    stream.write(f"  0  0 {nvar}\n")
    first_time = True
    m = 0
    for packet in _active_packets(combus):
        vid = packet.name
        for name in _packet_names(packet):
            if name == "time":
                if first_time:
                    first_time = False
                    stream.write(f"{'time':<{_FIELD_WIDTH}}")
                    m = 1
                continue
            buff = _truncate_label(name)
            if _is_vector_name(name):
                for n in range(1, 4):
                    stream.write(f"{buff}{n}_")
                    pad = 14 - len(buff)
                    stream.write(f"{vid:<{pad}}" if pad > 0 else vid)
                    m += 1
                    if m > 4:
                        m = 0
                        stream.write("\n")
            else:
                stream.write(f"{buff}_")
                pad = 15 - len(buff)
                stream.write(f"{vid:<{pad}}" if pad > 0 else vid)
                m += 1
                if m > 4:
                    m = 0
                    stream.write("\n")
    if nvar % _ACROSS:
        stream.write("\n")
    return nvar


def _flatten_packet_values(packet: Packet, *, skip_time: bool) -> list[float]:
    values: list[float] = []
    for name, value in packet.vars.items():
        if skip_time and name == "time":
            continue
        if _is_vector_name(name):
            values.extend(float(value[i]) for i in range(3))
        elif isinstance(value, int) and not isinstance(value, bool):
            values.append(float(value))
        else:
            values.append(float(value))
    return values


def write_traj_data(
    stream,
    combus: list[Packet],
    *,
    merge: bool = False,
) -> None:
    """Append one CADAC traj.asc data block (five-across, width 16)."""
    packets = _active_packets(combus)
    k = 0
    if merge:
        stream.write(f"{'-1.0':<{_FIELD_WIDTH}}")
        k = 1
    else:
        time_val = 0.0
        if packets and "time" in packets[0].vars:
            time_val = float(packets[0].vars["time"])
        elif combus and "time" in combus[0].vars:
            time_val = float(combus[0].vars["time"])
        stream.write(f"{time_val:<{_FIELD_WIDTH}}")
        k = 1
    for packet in packets:
        for value in _flatten_packet_values(packet, skip_time=True):
            if k > 4:
                k = 0
                stream.write("\n")
            stream.write(f"{value:<{_FIELD_WIDTH}}")
            k += 1
    stream.write("\n")


def open_traj_stream(out_dir: Path):
    """Open traj.asc for writing (C++ ofstream ftraj)."""
    path = Path(out_dir) / "traj.asc"
    stream = path.open("w", encoding="utf-8", newline="\n")
    return path, stream

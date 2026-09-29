"""CADAC console scrn_banner / scrn_data (y_scrn → stdout)."""

from __future__ import annotations

import sys

from cadac.io.tabout import collect_tabout_values, scrn_stems

# C++ scrn: label_length=14, field width 15, eight across (k>7).
_LABEL_LEN = 14
_FIELD_WIDTH = 15


def _truncate_label(name: str) -> str:
    return name[:_LABEL_LEN]


def _is_vector_stem(store, name: str) -> bool:
    field = store.field(name)
    if field.type == "vec":
        return True
    buff = _truncate_label(name)
    return bool(buff) and buff[0].isupper()


def write_scrn_banner(stream, vehicle_type: str, store) -> None:
    """Write CADAC scrn_banner header (Vehicle line + scrn labels) to stream."""
    stream.write(f"\n Vehicle: {vehicle_type} \n")
    k = 0
    for name in scrn_stems(store):
        buff = _truncate_label(name)
        if _is_vector_stem(store, name):
            for i in range(1, 4):
                pad = _FIELD_WIDTH - len(buff)
                stream.write(buff)
                stream.write(f"{i:<{pad}}" if pad > 0 else str(i))
                k += 1
                if k > 7:
                    k = 0
                    stream.write("\n")
        else:
            stream.write(f"{buff:<{_FIELD_WIDTH}}")
            k += 1
            if k > 7:
                k = 0
                stream.write("\n")
    stream.write("\n\n")


def write_scrn_data(stream, vehicle_name: str, store) -> None:
    """Append one CADAC scrn_data block (name + eight-across values)."""
    stream.write(f"{vehicle_name}\n")
    k = 0
    for value in collect_tabout_values(store):
        stream.write(f"{value:<{_FIELD_WIDTH}}")
        k += 1
        if k > 7:
            k = 0
            stream.write("\n")
    stream.write("\n")


def default_scrn_stream():
    """C++ cout target for y_scrn."""
    return sys.stdout

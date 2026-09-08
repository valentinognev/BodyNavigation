from collections.abc import Iterable
from dataclasses import dataclass

from cadac.kernel.state import StateStore


@dataclass
class Packet:
    name: str
    type: str
    status: int
    vars: dict


def packet_from_store(store: StateStore, names: Iterable[str]) -> Packet:
    return Packet(
        name="",
        type="",
        status=1,
        vars={name: store.get(name) for name in names},
    )

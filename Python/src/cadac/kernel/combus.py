from dataclasses import dataclass


@dataclass
class Packet:
    name: str
    type: str
    status: int
    vars: dict


def packet_from_store(store, names):
    return Packet(
        name="",
        type="",
        status=1,
        vars={name: store.get(name) for name in names},
    )

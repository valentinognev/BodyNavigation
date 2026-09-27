from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cadac.kernel.state import StateStore


@dataclass
class EventSpec:
    when: dict
    set: dict


class EventEngine:
    def __init__(self, events: Sequence[EventSpec]) -> None:
        self._events = list(events)
        self._index = 0

    def evaluate(self, store: StateStore) -> bool:
        if self._index >= len(self._events):
            return False
        event = self._events[self._index]
        if not _matches(store, event.when):
            return False
        for name, value in event.set.items():
            store.set(name, value)
        self._index += 1
        return True


def _parse_when(when: Mapping[str, Any]) -> tuple[str, str, Any]:
    if "var" in when and "op" in when and "value" in when:
        return when["var"], when["op"], when["value"]
    name, pred = next(iter(when.items()))
    op, value = next(iter(pred.items()))
    return name, op, value


def _matches(store: StateStore, when: Mapping[str, Any]) -> bool:
    name, op, crit = _parse_when(when)
    current = store.get(name)
    if type(current) is int:
        current = int(current)
        crit = int(crit)
    if op == "<":
        return current < crit
    if op == "=":
        return current == crit
    if op == ">":
        return current > crit
    return False

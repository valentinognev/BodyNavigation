from dataclasses import dataclass

import numpy as np


@dataclass
class Field:
    name: str
    value: object
    type: str
    role: str
    module: str
    outputs: tuple = ()


class StateStore:
    def __init__(self):
        self._fields = {}

    def define(self, field):
        if field.name in self._fields:
            raise ValueError(f"duplicate field {field.name!r}")
        field.value = self._coerce(field.type, field.value)
        self._fields[field.name] = field

    def __contains__(self, name):
        return name in self._fields

    def get(self, name):
        return self._fields[name].value

    def get_optional(self, name, default=None):
        field = self._fields.get(name)
        if field is None:
            return default
        return field.value

    def set(self, name, value):
        field = self._fields[name]
        field.value = self._coerce(field.type, value)

    def names(self):
        return list(self._fields)

    def field(self, name):
        return self._fields[name]

    def _coerce(self, ftype, value):
        if ftype == "int":
            if type(value) is int:
                return value
            return int(value)
        if ftype == "real":
            if type(value) is float:
                return value
            return float(value)
        if ftype == "vec":
            if (
                isinstance(value, np.ndarray)
                and value.shape == (3,)
                and value.dtype == float
            ):
                return value
            arr = np.asarray(value, dtype=float)
            if arr.shape != (3,):
                raise ValueError(f"vec must have shape (3,), got {arr.shape}")
            return arr
        if ftype == "mat":
            if (
                isinstance(value, np.ndarray)
                and value.shape == (3, 3)
                and value.dtype == float
            ):
                return value
            arr = np.asarray(value, dtype=float)
            if arr.shape != (3, 3):
                raise ValueError(f"mat must have shape (3, 3), got {arr.shape}")
            return arr
        raise ValueError(f"unknown type {ftype!r}")

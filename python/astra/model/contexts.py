"""Immutable categorical semantics with presentation-only display names."""
from dataclasses import dataclass, field
import unicodedata
import uuid


def _name(value):
    if type(value) is not str or not 1 <= len(value.strip()) <= 160 or any(unicodedata.category(c) == 'Cc' for c in value):
        raise ValueError('Context names require 1–160 visible characters')
    return value.strip()


def _id(value):
    if type(value) is not str or str(uuid.UUID(value)) != value:
        raise ValueError('Context IDs must be canonical UUIDs')
    return value


@dataclass(frozen=True)
class ContextValue:
    id: str
    name: str = field(compare=False)

    def to_dict(self):
        return {'id': self.id, 'name': self.name}


@dataclass(frozen=True)
class ContextField:
    id: str
    name: str = field(compare=False)
    values: tuple[ContextValue, ...]

    def to_dict(self):
        return {'id': self.id, 'name': self.name, 'values': [v.to_dict() for v in self.values]}

    def semantic_dict(self):
        return {'id': self.id, 'values': [v.id for v in self.values]}

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {'id', 'name', 'values'}:
            raise ValueError('Invalid context field')
        entries = value['values']
        if type(entries) is not list or len(entries) > 255:
            raise ValueError('A context field supports up to 255 authored values')
        values = []
        for item in entries:
            if type(item) is not dict or set(item) != {'id', 'name'}:
                raise ValueError('Invalid context value')
            values.append(ContextValue(_id(item['id']), _name(item['name'])))
        identifier = _id(value['id'])
        if len({v.id for v in values}) != len(values) or identifier in {v.id for v in values}:
            raise ValueError('Context value identities must be distinct')
        return cls(identifier, _name(value['name']), tuple(values))


def vocabulary(value, sizes):
    if type(value) is not list or len(value) != len(sizes) or len(value) > 32:
        raise ValueError('Context vocabulary must match the configured fields')
    result = tuple(ContextField.from_dict(item) for item in value)
    if len({f.id for f in result}) != len(result) or any(len(f.values) + 1 != size for f, size in zip(result, sizes)):
        raise ValueError('Context vocabulary must match its unknown-inclusive dimensions')
    return result

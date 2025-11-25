from __future__ import annotations

from types import UnionType
from typing import Iterator, TypeAlias, TypeVar

from typing_extensions import Self

from ._utils import class_and_instance_method

T = TypeVar("T")


_ClassInfo: TypeAlias = type | UnionType | tuple["_ClassInfo", ...]


class Node:
    """Node objects are descriptors with a name and a parent.

    Setting an instance attribute (__set__),
    sets that instance as the parent of that attribute,
    if it did not have a previous parent set.
    Hence, it must be an Node subclass.

    Attribute access from an instance
    tries to return the object in the instance's __dict__.
    Otherwise, it creates a copy with self._copy_from,
    assigns the instance as its parent
    and saves is in the instance's __dict__.
    Hence, Node subclasses must implement self._copy_from.
    """

    name: str = ""
    parent: Node | None = None

    def _copy_from(self, parent: Node) -> Self:
        raise NotImplementedError

    def __set_name__(self, cls: Node, name: str):
        if not (self.name == "" or self.name == name):
            raise NameError(f"cannot rename {self.name} to {name}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "parent", cls)

    def __set__(self, obj: Node, value: Self):
        if not isinstance(value, self.__class__):
            raise TypeError(f"unexpected type {type(value)} for {self.name}")

        if value.parent is None and value.name == "":
            # if it has no name, it was created outside an EagerNamer
            value.__set_name__(obj, self.name)

        obj.__dict__[self.name] = value

    def __get__(self, parent: Node | None, cls: type[Node]) -> Self:
        if parent is None:
            return self

        try:
            return parent.__dict__[self.name]
        except KeyError:
            copy = self._copy_from(parent)
            copy.__set_name__(parent, self.name)
            parent.__dict__[self.name] = copy
            return copy

    def __str__(self) -> str:
        # This is a recursive method, as self.parent is Owned | None
        if self.parent is None or self.parent.name == "":
            return self.name
        return f"{self.parent}.{self.name}"

    def __repr__(self):
        return str(self)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented

        return str(self) == str(other)

    @class_and_instance_method
    def _yield(
        self,
        type: type[T],
        /,
        *,
        exclude: _ClassInfo = (),
        recursive: bool = True,
    ) -> Iterator[T]:
        if isinstance(self, Node):
            cls = self.__class__
        else:
            cls = self

        for k, v in cls.__dict__.items():
            if isinstance(v, type) and not isinstance(v, exclude):
                yield getattr(self, k)

            if recursive and isinstance(v, Node):
                v: Node = getattr(self, k)
                yield from v._yield(type, exclude=exclude, recursive=recursive)


class NodeMapper:
    def __init__(self, obj: Node):
        self.obj = obj
        self.cls = obj.__class__

    def get(self, item: T, default: T | None = None) -> T:
        if not isinstance(item, Node):
            return item

        path = [item]
        while (item := item.parent) is not None:
            if isinstance(item, type) and issubclass(self.cls, item):
                item = self.obj
                for p in path[::-1]:
                    # It is important to use __get__ instead of getattr(item, p.name)
                    item = p.__get__(item, item.__class__)
                return item
            else:
                path.append(item)
        else:
            return item

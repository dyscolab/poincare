from collections import defaultdict
from collections.abc import Callable, Hashable, Iterator, Mapping
from types import MethodType, ModuleType
from typing import (
    Any,
    Concatenate,
    TypeVar,
    overload,
)
import logging

from symbolite import real
from symbolite.ops import count_named, substitute, translate


# TODO: add logger config
logger = logging.getLogger("poincare")


class class_and_instance_method[S, **P, R]:
    def __init__(self, func: Callable[Concatenate[S, P], R]):
        self.func = func

    @overload
    def __get__(self, obj: None, cls: type[S]) -> Callable[P, R]: ...

    @overload
    def __get__(self, obj: S, cls: type[S]) -> Callable[P, R]: ...

    def __get__(self, obj, cls):  # type: ignore
        if obj is None:
            return MethodType(self.func, cls)
        else:
            return MethodType(self.func, obj)


TH = TypeVar("TH", bound=Hashable)


def solve_dependencies(dependencies: Mapping[TH, set[TH]]) -> Iterator[set[TH]]:
    """Solve a dependency graph.

    Parameters
    ----------
    dependencies :
        dependency dictionary. For each key, the value is an iterable indicating its
        dependencies.

    Yields
    ------
    set
        iterator of sets, each containing keys of independents tasks dependent only of
        the previous tasks in the list.

    Raises
    ------
    ValueError
        if a cyclic dependency is found.
    """
    while dependencies:
        # values not in keys (items without dep)
        t = {i for v in dependencies.values() for i in v} - dependencies.keys()
        # and keys without value (items without dep)
        t.update(k for k, v in dependencies.items() if not v)
        # can be done right away
        if not t:
            raise ValueError(
                "Cyclic dependencies exist among these items: {}".format(
                    ", ".join(repr(x) for x in dependencies.items())
                )
            )
        # and cleaned up
        dependencies = {k: v - t for k, v in dependencies.items() if v}
        yield t


def eval_content[TH: Hashable](
    content: Mapping[TH, Any],
    libsl: ModuleType,
    is_root: Callable[[Any], bool],
    is_dependency: Callable[[Any], bool],
) -> dict[TH, real.NumberT]:
    out: dict[TH, real.NumberT] = {}

    dependencies = defaultdict(set)
    for k, v in content.items():
        if is_root(v):
            out[k] = v
        else:
            for el in filter(is_dependency, count_named(v).keys()):
                dependencies[k].add(el)

    layers = solve_dependencies(dependencies)

    for layer in layers:
        for item in layer:
            out[item] = translate(substitute(content[item], out), libsl)

    return out

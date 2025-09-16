from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

from symbolite.core import substitute

from ..compile import build_equation_maps
from ..types import Node, Symbol, System


def default_name(name) -> str:
    return f"\\text{{{name}}}".replace("_", "\\_")


@dataclass
class ToLatex:
    system: System | type[System]
    normalize_name: Callable[[Symbol], str] = default_name

    def __post_init__(self):
        self.equations = build_equation_maps(self.system)

    def yield_variables(self) -> Iterator[tuple[str, str, str]]:
        for x in self.equations.variables:
            name = self.normalize_name(x)
            yield name, str(x.initial)
            for order in range(1, x.equation_order):
                d = x.derivatives[order]
                yield (
                    self.normalize_name(d),
                    str(x.initial),
                    latex_derivative(name, order),
                )

    def yield_parameters(self) -> Iterator[tuple[str, str]]:
        for x in self.equations.parameters:
            yield self.normalize_name(x), str(x.default)

    def yield_equations(self) -> Iterator[tuple[str, str]]:
        try:
            from sympy.parsing.sympy_parser import parse_expr
            from sympy.printing import latex

            def normalize(eq):
                return latex(parse_expr(str(eq)))

        except ImportError:
            norm = Normalizer(self.normalize_name)

            def normalize(eq):
                return str(substitute(eq, norm))

        for der, eq in self.equations.func.items():
            d = latex_derivative(self.normalize_name(der.variable), der.order)
            eq = normalize(eq)
            yield d, eq

class Normalizer(dict):
    def __init__(self, func):
        self.func = func

    def get(self, key, default):
        if isinstance(key, Node):
            return self.func(key)
        return key


def as_aligned_lines(iterable, *, align_char: str):
    lines = []
    lines.append("\\begin{aligned}")
    lines.extend(yield_aligned(iterable, align_char=align_char))
    lines.append("\\end{aligned}")
    return "\n".join(lines)


def yield_aligned(
    iterable: Iterable[Iterable[str]],
    *,
    align_char: str = " & ",
) -> Iterable[str]:
    for x in iterable:
        yield align_char.join(x) + "\\\\"


def latex_derivative(name: str, order: int, with_respect_to: str = "t") -> str:
    if order == 1:
        return f"\\frac{{d{name}}}{{d{with_respect_to}}}"
    return f"\\frac{{d^{order}{name}}}{{d{with_respect_to}^{order}}}"

def latex_equations(model: System) -> str:
    return as_aligned_lines(ToLatex(model).yield_equations(), align_char= "=")

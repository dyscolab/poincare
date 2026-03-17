from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from io import StringIO, TextIOWrapper

from symbolite import Real
from symbolite.impl import liblatex
from symbolite.ops import substitute, translate, yield_named

from ..compile import build_equation_maps, replace_algebraic_equations
from ..types import (
    Constant,
    Derivative,
    Independent,
    Node,
    Parameter,
    System,
    Variable,
)

type Latex = str


def default_name(name: Real) -> Latex:
    return f"\\text{{{name}}}".replace("_", "\\_")


def escape_underscores(name: Real) -> Latex:
    return str(name).replace("_", "\\_")


def math_normalize(name: Real) -> Latex:
    string = str(name).replace("_", "_{")
    if str(name) != string:
        return string + "}"
    else:
        return string


@dataclass
class ToLatex:
    system: System | type[System]
    normalize_name: Callable[[Real], Latex] = default_name
    transform: dict[Real, str] = field(default_factory=dict)
    replace_algebraics: bool = False

    def __post_init__(self):
        if self.replace_algebraics:
            self.equations = replace_algebraic_equations(
                build_equation_maps(self.system)
            )
            self.func = self.equations.func
        else:
            self.equations = build_equation_maps(self.system)
            self.func = self.equations.func[0]

    def yield_variables(
        self, descriptions: dict[Real, str] | None = None
    ) -> Iterator[tuple[Latex, Latex, Latex] | tuple[Latex, Latex, Latex, Latex]]:
        for x in self.equations.variables:
            name = normalize_eq(x, transform=self.transform)
            if descriptions is not None:
                try:
                    yield (
                        name,
                        normalize(x.initial, transform=self.transform),
                        "-",
                        "\\text{" + descriptions[x] + "}",
                    )
                except KeyError:
                    yield name, normalize(x.initial, transform=self.transform), "-", "-"
            else:
                yield name, str(x.initial), "-"
            for order in range(1, x.equation_order):
                d = x.derivatives[order]
                if descriptions is not None:
                    try:
                        yield (
                            normalize_eq(d, transform=self.transform),
                            normalize(x.initial, transform=self.transform),
                            latex_derivative(name, order),
                            "\\text{" + descriptions[d] + "}",
                        )
                    except KeyError:
                        yield (
                            normalize_eq(d, transform=self.transform),
                            normalize(x.initial, transform=self.transform),
                            latex_derivative(name, order),
                            "-",
                        )
                else:
                    yield (
                        normalize_eq(d, transform=self.transform),
                        str(x.initial),
                        latex_derivative(name, order),
                    )

    def yield_parameters(
        self, descriptions: dict | None = None
    ) -> Iterator[tuple[Latex, Latex, Latex] | tuple[Latex, Latex]]:
        if descriptions is not None:
            for x in self.equations.parameters:
                yield (
                    normalize_eq(x, transform=self.transform),
                    str(x.default),
                    "\\text{" + descriptions.get(x, "-") + "}",
                )

        else:
            for x in self.equations.parameters:
                yield normalize_eq(x, transform=self.transform), str(x.default)

    def yield_equations(self) -> Iterator[tuple[Latex, Latex]]:
        for der, eq in self.func.items():
            d = latex_derivative(
                normalize_eq(der.variable, transform=self.transform), der.order
            )
            eq = normalize_eq(eq, self.transform)
            yield d, eq


class Normalizer(dict):
    def __init__(self, func):
        self.func = func

    def get(self, key, default=None):
        if isinstance(key, Node):
            return self.func(key)
        return key


def normalize(expr, transform: dict[Real, str]) -> Latex:
    if isinstance(expr, Real):
        return normalize_eq(expr, transform)
    else:
        return str(expr)


def normalize_eq(eq, transform) -> Latex:
    reps = {}
    real_transform = {key: Real(value) for key, value in transform.items()}
    for named in yield_named(eq):
        if isinstance(
            named, Independent | Constant | Parameter | Variable | Derivative
        ):
            reps[named] = real_transform.get(named, Real(named.name))
    eq = substitute(eq, reps)
    return translate(eq, liblatex).text


def as_aligned_lines(iterable, *, align_char: Latex):
    lines = []
    lines.append("\\begin{aligned}")
    lines.extend(yield_aligned(iterable, align_char=align_char))
    lines.append("\\end{aligned}")
    return "\n".join(lines)


def yield_aligned(
    iterable: Iterable[Iterable[Latex]],
    *,
    align_char: str = " & ",
) -> Iterable[Latex]:
    for x in iterable:
        yield align_char.join(x) + "\\\\"


def latex_derivative(name: str, order: int, with_respect_to: str = "t") -> Latex:
    if order == 1:
        return f"\\frac{{d{name}}}{{d{with_respect_to}}}"
    return f"\\frac{{d^{order}{name}}}{{d{with_respect_to}^{order}}}"


def latex_equations(
    model: type[System], transform: dict | None = None, latex: ToLatex | None = None
) -> Latex:
    if latex is None:
        transform = transform if transform is not None else {}
        latex = ToLatex(model, transform=transform)
    return as_aligned_lines(latex.yield_equations(), align_char="&=")


def parameter_table(
    model: type[System],
    transform: dict | None = None,
    descriptions: dict | None = None,
    latex: ToLatex | None = None,
) -> Latex:
    if latex is None:
        transform = transform if transform is not None else {}
        latex = ToLatex(model, transform=transform)
    parameters = latex.yield_parameters(descriptions=descriptions)

    if descriptions is not None:
        headers = ["Parameter", "Default", "Description"]
    else:
        headers = ["Parameter", "Default"]

    return make_latex_table(rows=parameters, headers=headers)


def variable_table(
    model: type[System],
    transform: dict | None = None,
    descriptions: dict | None = None,
    latex: ToLatex | None = None,
) -> Latex:
    if latex is None:
        transform = transform if transform is not None else {}
        latex = ToLatex(model, transform=transform)
    variables = latex.yield_variables(descriptions=descriptions)

    if descriptions is not None:
        headers = ["Variable", "Default", "Derivative", "Description"]
    else:
        headers = ["Variable", "Default", "Derivative"]

    return make_latex_table(rows=variables, headers=headers)


def make_latex_table(
    rows: Iterable[Iterable[Latex]], headers: Iterable[Latex]
) -> Latex:
    table = "\\begin{tabular}{|"
    table += len(headers) * "c|"
    table += "}\n"
    table += "\\hline\n"
    for head in headers:
        table += head + " & "
    table = table[:-2]
    table += "\\\\ \n \\hline \\hline \n"
    for row in rows:
        for element in row:
            table += "$" + element + "$" + " & "
        table = table[:-2]
        table += "\\\\ \n \\hline \n"
    table += "\\end{tabular}"
    return table


def make_model_report(
    model: type[System],
    report: TextIOWrapper | StringIO,
    transform: dict | None = None,
    descriptions: dict | None = None,
    standalone=True,
    replace_algebraics: bool = False,
):
    transform = transform if transform is not None else {}
    latex = ToLatex(
        system=model, transform=transform, replace_algebraics=replace_algebraics
    )
    if standalone:
        report.write(
            """\\documentclass{article}

\\usepackage{amsmath}
\\usepackage{float}

\\setcounter{secnumdepth}{0}

\\begin{document}"""
        )

    report.write(
        """\\subsection{Equations}

    """
    )
    report.write("\\[ " + latex_equations(model=model, latex=latex) + " \\]")

    report.write("""

    \\subsection{Variables}

    \\begin{table}[H]
     """)

    report.write(variable_table(model=model, latex=latex, descriptions=descriptions))
    report.write(
        """
    \\end{table}

    """
    )

    report.write(
        """
    \\subsection{Parameters}

    \\begin{table}[H]
    """
    )

    report.write(parameter_table(model=model, latex=latex, descriptions=descriptions))
    report.write(
        """
    \\end{table}

    """
    )

    if standalone:
        report.write("\\end{document}")

    return report


def model_report(
    model: type[System],
    path: str | None = None,
    transform: dict | None = None,
    descriptions: dict | None = None,
    standalone: bool = True,
    replace_algebraics: bool = False,
) -> Latex | None:
    open_form = "w" if standalone else "a"
    if path is None:
        write_to = StringIO("")
        return make_model_report(
            model=model,
            report=write_to,
            transform=transform,
            descriptions=descriptions,
            standalone=standalone,
            replace_algebraics=replace_algebraics,
        ).getvalue()

    else:
        write_to = path
        with open(write_to, open_form, encoding="utf-8") as report:
            make_model_report(
                model=model,
                report=report,
                transform=transform,
                descriptions=descriptions,
                standalone=standalone,
                replace_algebraics=replace_algebraics,
            )

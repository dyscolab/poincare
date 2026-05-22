from collections.abc import Callable, Iterable, Iterator, Mapping
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


def parent_path(named: Node, base: Node, path):
    parent = named.parent
    if parent == base:
        return path
    else:
        return parent_path(
            parent, base, parent.name + "․" + path
        )  # One dot leader, not period because it confuses attrgetter in symbolite


@dataclass
class ToLatex:
    system: System | type[System]
    normalize_name: Callable[[Real], Latex] = default_name
    transform: dict[Real, str] = field(default_factory=dict)
    descriptions: dict[Real, str] | None = None
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
        self,
    ) -> Iterator[tuple[Latex, Latex, Latex] | tuple[Latex, Latex, Latex, Latex]]:
        for x in self.equations.variables:
            name = normalize_eq(x, transform=self.transform, base=self.system)
            if self.descriptions is not None:
                try:
                    yield (
                        name,
                        normalize(
                            x.initial, transform=self.transform, base=self.system
                        ),
                        "-",
                        "\\text{" + self.descriptions[x] + "}",
                    )
                except KeyError:
                    yield (
                        name,
                        normalize(
                            x.initial, transform=self.transform, base=self.system
                        ),
                        "-",
                        "-",
                    )
            else:
                yield name, str(x.initial), "-"
            for order in range(1, x.equation_order):
                d = x.derivatives[order]
                if self.descriptions is not None:
                    try:
                        yield (
                            normalize_eq(d, transform=self.transform, base=self.system),
                            normalize(
                                x.initial, transform=self.transform, base=self.system
                            ),
                            latex_derivative(name, order),
                            "\\text{" + self.descriptions[d] + "}",
                        )
                    except KeyError:
                        yield (
                            normalize_eq(d, transform=self.transform, base=self.system),
                            normalize(
                                x.initial, transform=self.transform, base=self.system
                            ),
                            latex_derivative(name, order),
                            "-",
                        )
                else:
                    yield (
                        normalize_eq(d, transform=self.transform, base=self.system),
                        str(x.initial),
                        latex_derivative(name, order),
                    )

    def yield_parameters(
        self,
    ) -> Iterator[tuple[Latex, Latex, Latex] | tuple[Latex, Latex]]:
        if self.descriptions is not None:
            for x in self.equations.parameters:
                yield (
                    normalize_eq(x, transform=self.transform, base=self.system),
                    str(x.default),
                    "\\text{" + self.descriptions.get(x, "-") + "}",
                )

        else:
            for x in self.equations.parameters:
                yield (
                    normalize_eq(x, transform=self.transform, base=self.system),
                    str(x.default),
                )

    def yield_equations(self) -> Iterator[tuple[Latex, Latex]]:
        for der, eq in self.func.items():
            d = latex_derivative(
                normalize_eq(der.variable, transform=self.transform, base=self.system),
                der.order,
            )
            eq = normalize_eq(eq, self.transform, base=self.system)
            yield d, eq


class Normalizer(dict):
    def __init__(self, func):
        self.func = func

    def get(self, key, default=None):
        if isinstance(key, Node):
            return self.func(key)
        return key


def normalize(expr, transform: dict[Real, str], base: Node) -> Latex:
    if isinstance(expr, Real):
        return normalize_eq(expr, transform, base)
    else:
        return str(expr)


def normalize_eq(eq, transform, base: Node) -> Latex:
    reps = {}
    real_transform = {key: Real(value) for key, value in transform.items()}
    for named in yield_named(eq):
        if isinstance(
            named, Independent | Constant | Parameter | Variable | Derivative
        ):
            reps[named] = real_transform.get(
                named, Real(parent_path(named, base, named.name))
            )
    eq = substitute(eq, reps)
    return translate(eq, liblatex).text.replace(
        "․",
        ".",
    )  # Replace one dot leaders added by parent_path by regular periods.


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
    return "\\[ " + as_aligned_lines(latex.yield_equations(), align_char="&=") + " \\]"


def parameter_table(
    model: type[System],
    transform: dict | None = None,
    descriptions: dict | None = None,
    latex: ToLatex | None = None,
) -> Latex:
    if latex is None:
        transform = transform if transform is not None else {}
        latex = ToLatex(model, transform=transform, descriptions=descriptions)
    parameters = latex.yield_parameters()

    if latex.descriptions is not None:
        headers = ["Parameter", "Default", "Description"]
    else:
        headers = ["Parameter", "Default"]

    return (
        "\\begin{table}[H]\n\\centering\n"
        + make_latex_table(rows=parameters, headers=headers)
        + "\n\\end{table}"
    )


def variable_table(
    model: type[System],
    transform: dict | None = None,
    descriptions: dict | None = None,
    latex: ToLatex | None = None,
) -> Latex:
    if latex is None:
        transform = transform if transform is not None else {}
        latex = ToLatex(model, transform=transform, descriptions=descriptions)
    variables = latex.yield_variables()

    if latex.descriptions is not None:
        headers = ["Variable", "Default", "Derivative", "Description"]
    else:
        headers = ["Variable", "Default", "Derivative"]

    return (
        "\\begin{table}[H]\n\\centering\n"
        + make_latex_table(rows=variables, headers=headers)
        + "\n\\end{table}"
    )


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
    transform: dict | None,
    descriptions: dict | None,
    standalone,
    replace_algebraics: bool,
    sections: Mapping[str, Callable[[System, ToLatex], str]],
    packages: Iterable[str],
):
    transform = transform if transform is not None else {}
    latex = ToLatex(
        system=model,
        transform=transform,
        descriptions=descriptions,
        replace_algebraics=replace_algebraics,
    )
    if standalone:
        report.write(
            """\\documentclass{article}

"""
        )
        for package in packages:
            report.write(f"\\usepackage{{{package}}}\n")
        report.write("\\usepackage[margin=2cm]{geometry}\n")
        report.write("""\\setcounter{secnumdepth}{0}

\\begin{document}
""")
    for title, writer in sections.items():
        report.write(f"\\subsection{{{title}}}" + "\n")
        report.write(writer(model, latex=latex) + "\n\n")
    if standalone:
        report.write("\\end{document}")

    return report


default_sections = {
    "Equations": latex_equations,
    "Variables": variable_table,
    "Parameters": parameter_table,
}
default_packages = ["amsmath", "float"]


def model_report(
    model: type[System],
    path: str | None = None,
    transform: dict | None = None,
    descriptions: dict | None = None,
    standalone: bool = True,
    replace_algebraics: bool = False,
    sections: Mapping[str, Callable[[System, ToLatex], str]] = default_sections,
    packages: Iterable[str] = default_packages,
) -> Latex | None:
    open_form = "w" if standalone else "a"
    if path is None:
        with StringIO("") as write_to:
            return make_model_report(
                model=model,
                report=write_to,
                transform=transform,
                descriptions=descriptions,
                standalone=standalone,
                replace_algebraics=replace_algebraics,
                sections=sections,
                packages=packages,
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
                sections=sections,
                packages=packages,
            )

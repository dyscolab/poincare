from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

from symbolite.core import substitute

from ..compile import build_equation_maps
from ..types import Node, Symbol, System
from ..printing.table import Table
from io import StringIO, TextIOWrapper

type Latex = str


def default_name(name: Symbol) -> Latex:
    return f"\\text{{{name}}}".replace("_", "\\_")


def escape_underscores(name: Symbol) -> Latex:
    return str(name).replace("_", "\\_")


def math_normalize(name: Symbol) -> Latex:
    string = str(name).replace("_", "_{")
    if str(name) != string:
        return string + "}"
    else:
        return string


@dataclass
class ToLatex:
    system: System | type[System]
    normalize_name: Callable[[Symbol], Latex] = default_name
    transform: dict[Symbol, str] = field(default_factory=dict)

    def __post_init__(self):
        self.equations = build_equation_maps(self.system)

    def yield_variables(
        self, descriptions: dict[Symbol, str] | None = None
    ) -> Iterator[tuple[Latex, Latex, Latex] | tuple[Latex, Latex, Latex, Latex]]:
        for x in self.equations.variables:
            name = normalize_eq(x, transform=self.transform)
            if descriptions != None:
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
                if descriptions != None:
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
        if descriptions != None:
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
        for der, eq in self.equations.func.items():
            d = latex_derivative(
                normalize_eq(der.variable, transform=self.transform), der.order
            )
            eq = normalize_eq(eq, self.transform)
            yield d, eq


class Normalizer(dict):
    def __init__(self, func):
        self.func = func

    def get(self, key, default):
        if isinstance(key, Node):
            return self.func(key)
        return key


def normalize(expr, transform: dict[Symbol, str]) -> Latex:
    if isinstance(expr, Symbol):
        return normalize_eq(expr, transform)
    else:
        return str(expr)


def normalize_eq(eq, transform) -> Latex:
    from poincare import Parameter, Constant, Variable, Independent, Derivative
    from symbolite import Scalar

    reps = {}
    for named in eq.yield_named():
        if isinstance(
            named, Independent | Constant | Parameter | Variable | Derivative
        ):
            reps[named] = Scalar(named.name)
    eq = eq.subs(reps)
    try:
        from symbolite.impl import libsympy
        from sympy import latex
        import sympy as smp

        smp_exp = eq.eval(libsl=libsympy)
        for symb, trans in transform.items():
            smp_exp = smp_exp.subs(smp.Symbol(str(symb)), smp.Symbol(trans))
        return latex(smp.simplify(smp_exp))
    except ImportError:
        raise Exception("This function requires Sympy")


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


def latex_equations(model: type[System], transform: dict | None = None) -> Latex:
    transform = transform if transform is not None else {}
    return as_aligned_lines(
        ToLatex(model, transform=transform).yield_equations(), align_char="&="
    )


def parameter_table(
    model: type[System], transform: dict | None = None, descriptions: dict | None = None
) -> Latex:
    transform = transform if transform is not None else {}
    latex = ToLatex(model, transform=transform)
    parameters = latex.yield_parameters(descriptions=descriptions)

    if descriptions != None:
        headers = ["Parameter", "Default", "Description"]
    else:
        headers = ["Parameter", "Default"]

    return make_latex_table(rows=parameters, headers=headers)


def varaible_table(
    model: type[System], transform: dict | None = None, descriptions: dict | None = None
) -> Latex:
    transform = transform if transform is not None else {}
    latex = ToLatex(model, transform=transform)
    variables = latex.yield_variables(descriptions=descriptions)

    if descriptions != None:
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
):
    # if standalone:
    #     report.write("""\\documentclass{article}

    #     \\usepackage{amsmath}
    #     \\usepackage{float}

    #     \\setcounter{secnumdepth}{0}

    #     \\begin{document}
    #     """)

    # report.write("""
    # \\subsection{Equations}

    # """)
    # report.write("\\[ " + latex_equations(model=model, transform=transform) + " \\]")

    # report.write("""

    # \\subsection{Variables}

    # \\begin{table}[H]
    # """)
    # report.write(
    #     varaible_table(model=model, transform=transform, descriptions=descriptions)
    # )
    # report.write("""
    # \\end{table}

    # """)

    # report.write("""
    # \\subsection{Parameters}

    # \\begin{table}[H]
    # """)

    # report.write(
    #     parameter_table(model=model, transform=transform, descriptions=descriptions)
    # )
    # report.write("""
    # \\end{table}

    # """)

    if standalone:
        report.write("""\\documentclass{article}

        \\usepackage{amsmath}
        \\usepackage{float}

        \\setcounter{secnumdepth}{0}

        \\begin{document}
        """)

    report.write("""
    \\subsection{Equations}

    """)
    report.write("\\[ " + latex_equations(model=model, transform=transform) + " \\]")

    report.write("""

    \\subsection{Variables}

    \\begin{table}[H]
     """)

    report.write(
        varaible_table(model=model, transform=transform, descriptions=descriptions)
    )
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

    report.write(
        parameter_table(model=model, transform=transform, descriptions=descriptions)
    )
    report.write(
        """
    \\end{table}

    """
    )

    if standalone:
        report.write("\\end{document}")

    # Printing to file:

    # if standalone:
    #     print(
    #         """\\documentclass{article}

    #     \\usepackage{amsmath}
    #     \\usepackage{float}

    #     \\setcounter{secnumdepth}{0}

    #     \\begin{document}
    #     """,
    #         file=report,
    #     )

    # print(
    #     """
    # \\subsection{Equations}

    # """,
    #     file=report,
    # )
    # print(
    #     "\\[ " + latex_equations(model=model, transform=transform) + " \\]", file=report
    # )

    # print(
    #     """

    # \\subsection{Variables}

    # \\begin{table}[H]
    # """,
    #     file=report,
    # )

    # print(
    #     varaible_table(model=model, transform=transform, descriptions=descriptions),
    #     file=report,
    # )
    # print(
    #     """
    # \\end{table}

    # """,
    #     file=report,
    # )

    # print(
    #     """
    # \\subsection{Parameters}

    # \\begin{table}[H]
    # """,
    #     file=report,
    # )

    # print(
    #     parameter_table(model=model, transform=transform, descriptions=descriptions),
    #     file=report,
    # )
    # print(
    #     """
    # \\end{table}

    # """,
    #     file=report,
    # )

    # if standalone:
    #     print("\\end{document}", file=report)

    return report


def model_report(
    model: type[System],
    path: str | None = None,
    transform: dict | None = None,
    descriptions: dict | None = None,
    standalone=True,
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
            )

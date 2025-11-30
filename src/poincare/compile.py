from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, MutableSequence, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import (
    Any,
    Literal,
    Never,
    Protocol,
    TypeVar,
)

import pint
import symbolite.abstract as libabstract
from symbolite import real, vector
from symbolite.impl import libpythoncode
from symbolite.ops import substitute, translate, yield_named

from ._node import Node
from ._utils import eval_content
from .types import (
    Constant,
    Derivative,
    Equation,
    EquationGroup,
    Independent,
    Initial,
    Number,
    Parameter,
    System,
    Variable,
)

_symbolite_compile: (
    Callable[[str, ModuleType], Mapping[str, Callable[..., Any]]] | None
) = None


def symbolite_compile(
    source: str, libsl: ModuleType
) -> Mapping[str, Callable[..., Any]]:
    global _symbolite_compile
    if _symbolite_compile is None:
        from symbolite.impl._lang_value_utils import compile as _compile

        _symbolite_compile = _compile
    return _symbolite_compile(source, libsl)


V = TypeVar("V")
F = TypeVar("F")
type ExprRHS = Initial | real.Real
type Array = Sequence[float]
type MutableArray = MutableSequence[float]


class RHS(Protocol):
    def __call__(self, t: float, y: Array, p: Array, dy: MutableArray) -> Array: ...


class Transform(Protocol):
    def __call__(self, t: float, y: Array, p: Array, out: MutableArray) -> Array: ...


@dataclass(frozen=True, kw_only=True)
class Compiled[V, F]:
    independent: Sequence[Independent]
    variables: Sequence[V]
    parameters: Sequence[real.Real]
    mapper: dict[real.Real, Any]
    func: F
    output: dict[str, ExprRHS]
    libsl: ModuleType | None = None


def identity(x):
    return x


Backend = Literal["numpy", "numba", "jax"]


def get_libsl(backend: Backend) -> ModuleType:
    match backend:
        case "numpy":
            from symbolite.impl import libnumpy

            return libnumpy
        case "numba":
            from symbolite.impl import libnumpy

            return libnumpy
        case "jax":
            from symbolite.impl import libjax

            return libjax
        case _:
            assert_never(backend, message="Unknown backend {}")


def eqsum(eqs: list[ExprRHS]) -> real.NumberT | real.Real | ExprRHS:
    if len(eqs) == 0:
        return 0
    elif len(eqs) == 1:
        return eqs[0]
    else:
        return sum(eqs[1:], start=eqs[0])


def vector_mapping(
    time: real.Real,
    variables: Sequence[Variable | Derivative],
    parameters: Sequence[Parameter],
    time_varname: str = "t",
    state_varname: str = "y",
    param_varname: str = "p",
) -> dict[real.Real | Variable | Derivative | Parameter, real.Real]:
    t = real.Real(time_varname)
    y = vector.Vector(state_varname)
    p = vector.Vector(param_varname)
    mapping: dict[real.Real | Parameter | Variable | Derivative, real.Real] = {
        time: t,
    }
    for i, v in enumerate(variables):
        mapping[v] = y[i]
    for i, v in enumerate(parameters):
        mapping[v] = p[i]
    return mapping


def yield_equations(system: System | type[System]) -> Iterator[Equation]:
    for v in system._yield(Equation | EquationGroup):
        if isinstance(v, Equation):
            yield v
        elif isinstance(v, EquationGroup):
            yield from v.equations
        else:
            assert_never(v, message="unexpected type {}")


def get_equations(system: System | type[System]) -> dict[Derivative, list[ExprRHS]]:
    equations: dict[Derivative, list[ExprRHS]] = defaultdict(list)
    for eq in yield_equations(system):
        equations[eq.lhs].append(eq.rhs)
    return equations


def depends_on_at_least_one_variable_or_time(value: Any) -> bool:
    if not isinstance(value, real.Real):
        return False

    for named in yield_named(value):
        if isinstance(named, Independent):
            return True
        elif isinstance(named, Variable | Derivative):
            return True
        elif isinstance(named, Parameter) and depends_on_at_least_one_variable_or_time(
            named.default
        ):
            return True
    return False


def get_derivative(variable: Variable, order: int) -> Variable | Derivative:
    if order == 0:
        return variable
    try:
        return variable.derivatives[order]
    except KeyError:
        return Derivative(variable, order=order)


def build_equation_maps(
    system: System | type[System],
) -> Compiled[Variable, dict[Derivative, ExprRHS]]:
    """Compiles equations into dicts of equations.

    - variables: Variable | Derivative
        appears in one or more equations
    - parameters: Parameter
        appears in one or more equations and is not a function of time or variables
    - algebraic_equations: dict[Parameter, RHSExpr]
        parameters that are functions of time or variables
    - differential_equations: dict[Variable | Derivative, RHSExpr]
        variables whose differential are functions of time or variables
    """

    algebraic: dict[Parameter, ExprRHS] = {}
    equations: dict[Derivative, ExprRHS] = {
        k: eqsum(v) for k, v in get_equations(system).items()
    }

    initials = {}
    independent: set[Independent] = set()
    parameters: set[Parameter] = set()
    variables: set[Variable] = set()

    def add_to_initials(name: real.Real, value):
        if value is None:
            raise TypeError(
                f"Missing initial value for {name}. System must be instantiated."
            )
        initials[name] = value
        if not isinstance(value, real.Real):
            return

        for named in yield_named(value):
            if isinstance(named, Independent):
                independent.add(named)
            elif isinstance(named, Parameter | Constant):
                add_to_initials(named, named.default)

    def process_symbol(symbol, *, equation: bool):
        if not isinstance(symbol, real.Real):
            return

        # TODO: it would be good to document this in a highlevel manner
        for named in yield_named(symbol):
            if isinstance(named, Independent):
                independent.add(named)
            elif isinstance(named, Variable):
                if named.equation_order is None:
                    if equation:
                        # TODO: Why is this here Would it make sense to cast it to parameter?
                        parameters.add(named)
                    add_to_initials(named, named.initial)
                else:
                    if equation:
                        variables.add(named)
                    for order in range(named.equation_order):
                        der = get_derivative(named, order)
                        add_to_initials(der, der.initial)
            elif isinstance(
                named, Parameter
            ) and depends_on_at_least_one_variable_or_time(named.default):
                algebraic[named] = named.default
                process_symbol(named.default, equation=equation)
                add_to_initials(named, named.default)
            elif isinstance(named, Constant | Parameter):
                # TODO: Why is this here Would it make sense to cast it to parameter?
                if equation:
                    parameters.add(named)
                add_to_initials(named, named.default)

    for derivative, eq in equations.items():
        process_symbol(derivative.variable, equation=True)
        process_symbol(eq, equation=True)
    for symbol in system._yield(Independent | Constant | Parameter | Variable):
        process_symbol(symbol, equation=False)

    match len(independent):
        case 0:
            time = Independent(default=0)
        case 1:
            time = independent.pop()
        case _:
            raise TypeError(f"more than one independent variable found: {independent}")

    root = {time, *variables, *parameters}
    for v in variables:
        if v.equation_order is not None:
            root.update(v.derivatives[order] for order in range(1, v.equation_order))

    def is_root(x):
        if isinstance(x, Number | pint.Quantity):
            return True
        elif x in root:
            return True
        else:
            return False

    content = {
        **initials,
        **equations,
        **algebraic,
        **{x: x for x in root},
    }

    content = eval_content(
        content,
        libabstract,
        is_root=is_root,
        is_dependency=lambda x: isinstance(x, Node),
    )

    equations = {k: content[k] for k in equations.keys()}
    sorted_variables = sorted(variables, key=str)
    return Compiled(
        independent=(time,),
        variables=sorted_variables,
        parameters=sorted(parameters, key=str),
        mapper=initials,
        func=equations,
        output=sorted_variables,
    )


def build_first_order_symbolic_ode(
    system: System | type[System],
) -> Compiled[Variable | Derivative, dict[Variable | Derivative, ExprRHS]]:
    maps = build_equation_maps(system)

    # Differential equations
    # Map variable to be derived 1 time to equation.
    # (unlike 'equations' that maps derived variable to equation)
    variables: list[Variable | Derivative] = []
    diff_eqs: dict[Variable | Derivative, ExprRHS] = {}
    for var in maps.variables:
        var: Variable
        # For each variable
        # - create first order differential equations except for var.equation_order
        # - for the var.equation_order use the defined equation
        if var.equation_order is None:
            raise TypeError

        for order in range(var.equation_order - 1):
            lhs = get_derivative(var, order)
            rhs = get_derivative(var, order + 1)
            diff_eqs[lhs] = rhs
            variables.append(lhs)

        order = var.equation_order
        lhs = get_derivative(var, order - 1)
        rhs = get_derivative(var, order)
        diff_eqs[lhs] = maps.func[rhs]
        variables.append(lhs)

    return Compiled(
        independent=maps.independent,
        variables=variables,
        output={str(v): v for v in variables},
        func=diff_eqs,
        parameters=maps.parameters,
        mapper=maps.mapper,
    )


def assignment(name: str, index: str, value: str) -> str:
    return f"{name}[{index}] = {value}"


def jax_assignment(name: str, index: str, value: str) -> str:
    return f"{name} = {name}.at[{index}].set({value})"


def build_first_order_vectorized_body(
    system: System | type[System],
    *,
    assignment_func: Callable[[str, str, str], str] = assignment,
) -> Compiled[Variable | Derivative, str]:
    symbolic = build_first_order_symbolic_ode(system)
    mapping: Mapping = vector_mapping(
        symbolic.independent[0],
        symbolic.variables,
        symbolic.parameters,
    )

    diff_eqs = {k: substitute(v, mapping) for k, v in symbolic.func.items()}

    def to_index(k: str) -> str:
        try:
            return str(symbolic.variables.index(k))
        except ValueError:
            pass

        try:
            return str(symbolic.parameters.index(k))
        except ValueError:
            pass

        raise ValueError(k)

    ode_step_def = "\n    ".join(
        [
            "def ode_step(t, y, p, ydot):",
            *(
                assignment_func("ydot", to_index(k), translate(eq, libpythoncode))
                for k, eq in diff_eqs.items()
            ),
            "return ydot",
        ]
    )

    return Compiled(
        independent=symbolic.independent,
        variables=symbolic.variables,
        parameters=symbolic.parameters,
        mapper=symbolic.mapper,
        func=ode_step_def,
        output=symbolic.output,
    )


def build_first_order_functions(
    system: System | type[System],
    libsl: ModuleType,
    optimizer: Callable[[RHS], RHS] = identity,
    assignment_func: Callable[[str, str, str], str] = assignment,
) -> Compiled[Variable | Derivative, RHS]:
    vectorized = build_first_order_vectorized_body(
        system, assignment_func=assignment_func
    )
    lm = symbolite_compile(vectorized.func, libsl)
    return Compiled(
        independent=vectorized.independent,
        variables=vectorized.variables,
        parameters=vectorized.parameters,
        mapper=vectorized.mapper,
        func=optimizer(lm["ode_step"]),
        output=vectorized.output,
        libsl=libsl,
    )


def compile_diffeq(
    system: System | type[System],
    backend: Backend,
) -> Compiled[Variable | Derivative, RHS]:
    libsl = get_libsl(backend)

    optimizer_fun = identity
    assignment_fun = assignment
    match backend:
        case "numba":
            import numba

            optimizer_fun = numba.njit
        case "jax":
            import jax

            optimizer_fun = jax.jit
            assignment_fun = jax_assignment

    return build_first_order_functions(system, libsl, optimizer_fun, assignment_fun)


def assert_never(arg: Never, *, message: str) -> Never:
    raise ValueError(message.format(arg))


def identity_transform(t: float, y: Array, p: Array, out: MutableArray) -> Array:
    out[:] = y
    return out


def compile_transform(
    system: System | type[System],
    compiled: Compiled,
    expresions: Mapping[str, real.Real] | None = None,
) -> Compiled[Variable | Derivative, Transform]:
    assert compiled.libsl is not None

    if expresions is None:
        return Compiled(
            func=identity_transform,
            output=compiled.output,
            independent=compiled.independent,
            variables=compiled.variables,
            parameters=compiled.parameters,
            mapper=compiled.mapper,
            libsl=compiled.libsl,
        )

    root = {
        x: x
        for x in (
            *compiled.independent,
            *compiled.variables,
            *compiled.parameters,
        )
    }

    def is_root(x):
        if isinstance(x, Number | pint.Quantity):
            return True
        elif x in root:
            return True
        else:
            return False

    content = {
        **expresions,
        **compiled.mapper,
        **root,
    }
    content = eval_content(
        content,
        libabstract,
        is_root=is_root,
        is_dependency=lambda x: isinstance(x, Node),
    )
    content_in_expresions = {k: content[k] for k in expresions}

    mapping: Mapping = vector_mapping(
        compiled.independent[0],
        compiled.variables,
        compiled.parameters,
    )

    def to_index(k: str) -> str:
        try:
            return str(compiled.variables.index(k))
        except ValueError:
            pass

        try:
            return str(compiled.parameters.index(k))
        except ValueError:
            pass

        raise ValueError(k)

    deqs = {k: substitute(v, mapping) for k, v in content_in_expresions.items()}
    ode_step_def = "\n    ".join(
        [
            "def transform(t, y, p, out):",
            *(
                assignment("out", str(i), translate(eq, libpythoncode))
                for i, eq in enumerate(deqs.values())
            ),
            "return out",
        ]
    )
    lm = symbolite_compile(ode_step_def, compiled.libsl)
    return Compiled(
        func=lm["transform"],
        output=content_in_expresions,
        independent=compiled.independent,
        variables=compiled.variables,
        parameters=compiled.parameters,
        mapper=compiled.mapper,
        libsl=compiled.libsl,
    )

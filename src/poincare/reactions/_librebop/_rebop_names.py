import numpy as np
from symbolite.abstract import real as abstract_real
from symbolite.core.function import (
    BinaryFunction,
    BinaryOperator,
    UnaryFunction,
    UnaryOperator,
)

latex_names = {
    abstract_real.add: "{} + {}",
    abstract_real.sub: "{} - {}",
    abstract_real.mul: "{} * {}",
    abstract_real.truediv: "{} / {}",
    abstract_real.neg: "-{}",
    abstract_real.pos: "+{}",
    abstract_real.degrees: "{} *" + f"180/{str(np.pi)}",
    abstract_real.exp: f"{str(np.e)}" + "^{}",
    abstract_real.hypot: "({}^2 + {}^2)^0.5",
    abstract_real.radians: "{} *" + f"{str(np.pi)} */180",
    abstract_real.sqrt: "({}^0.5)",
    abstract_real.pow: "{}^{}",
}


def get_rebop_name(
    obj: UnaryFunction | BinaryFunction | UnaryOperator | BinaryOperator,
) -> str:
    return latex_names[obj]


rebop_precedences = {
    abstract_real.sqrt: 4,
    abstract_real.hypot: 2,
    abstract_real.pow: 3,
    abstract_real.exp: 3,
    abstract_real.degrees: 2,
    abstract_real.radians: 2,
}


def get_rebop_precedence(
    obj: UnaryFunction | BinaryFunction | UnaryOperator | BinaryOperator,
) -> int:
    """Some functions are turned into operators for latex conversion
    or given a different precedence."""
    precedence = rebop_precedences.get(
        obj, getattr(obj.__symbolite_info__, "precedence", None)
    )
    if precedence is None:
        raise AttributeError(f"{obj} has no precedence")
    else:
        return precedence

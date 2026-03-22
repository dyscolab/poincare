"""
symbolite.impl.libpythoncode.real
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Code-emitting counterparts for ``symbolite.abstract.real``.

:copyright: 2023 by Symbolite Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import numpy as np
from symbolite.abstract import real as abstract_real

from ._rebopexpr import (
    RebopExpr,
    as_operator,
)


def Real(name: str) -> RebopExpr:
    return RebopExpr(name.replace(".", "__"))


add = as_operator(abstract_real.add)
sub = as_operator(abstract_real.sub)
mul = as_operator(abstract_real.mul)
truediv = as_operator(abstract_real.truediv)
neg = as_operator(abstract_real.neg)
pos = as_operator(abstract_real.pos)
degrees = as_operator(abstract_real.degrees)
exp = as_operator(abstract_real.exp)
hypot = as_operator(abstract_real.hypot)
radians = as_operator(abstract_real.radians)
sqrt = as_operator(abstract_real.sqrt)
pow = as_operator(abstract_real.pow)


e = RebopExpr(str(np.e))
pi = RebopExpr(str(np.pi))
tau = RebopExpr(str(2 * np.pi))

__all__ = [
    "Real",
    "add",
    "sub",
    "mul",
    "neg",
    "pos",
    "degrees",
    "exp",
    "radians",
    "sqrt",
    "pow",
    "e",
    "pi",
    "tau",
]

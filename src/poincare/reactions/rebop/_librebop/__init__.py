"""
symbolite implementation to translate a subset of expressions into the string format required for rebop
"""

from symbolite.impl import Kind

from . import lang, real

KIND = Kind.CODE


__all__ = ["real", "lang"]

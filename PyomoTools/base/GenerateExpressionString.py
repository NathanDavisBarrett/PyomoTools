import pyomo.environ as pyo
from pyomo.core.expr import identify_variables


def GenerateExpressionStrings(expr):
    """
    A function to generate a pair of string representations of a pyomo expression. The first will be the original, symbolic pyomo expression string (e.g. what you'd get from calling str(expr) but with some added spaces). The second is the same string but with each variable replaced with it's corresponding value.

    Parameters
    ----------
    expr: pyomo expression object
        The expression you'd like to generate a substituted string for.

    Returns
    -------
    tuple of str:
        symStr: The symbolic expression string
        numStr: The numeric (substituted) string
    """
    if expr is None:
        return "", ""

    symStr = str(expr)
    numStr = str(expr)

    vrs = list(identify_variables(expr))
    vrs = sorted(vrs, reverse=True, key=lambda v: len(str(v)))

    for v in vrs:
        varStr = v.name if hasattr(v, "name") else v.getname()
        try:
            valStr = str(pyo.value(v))
        except ValueError as e:
            raise ValueError(
                f"The following error was thrown when attempting to collect the value of {varStr}:\n{e}"
            )

        varStrLen = len(varStr)
        valStrLen = len(valStr)

        if varStrLen >= valStrLen:
            valStr = valStr + " " * (varStrLen - valStrLen)
            numStr = numStr.replace(varStr, valStr)
        else:
            newVarStr = varStr + " " * (valStrLen - varStrLen)
            numStr = numStr.replace(varStr, valStr)
            symStr = symStr.replace(varStr, newVarStr)

    return symStr, numStr

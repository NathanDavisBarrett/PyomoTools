import pyomo.environ as pyo
from pyomo.core.expr.current import identify_variables

from .MergeableModel import MergableModel

def MergeableModelFilter(s:str):
    """
    A function to convert a given string containing the somewhat un-humanly readable variable/constraint names created by MergableModels to a more readable version.

    Parameters
    ----------
    s: str 
        The string you'd like to convert

    Returns
    -------
    s: str
        The converted string
    """
    return s.replace(MergableModel.subModelKeyword,'').replace(MergableModel.componentKeyword,'.')

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
    symStr = str(expr)
    numStr = str(expr)

    vrs = list(identify_variables(expr))
    vrs = sorted(vrs,reverse=True,key=lambda v:len(str(v)))

    for v in vrs:
        varStr = v.getname()
        valStr = str(pyo.value(v))

        varStrLen = len(varStr)
        valStrLen = len(valStr)

        if varStrLen >= valStrLen:
            valStr = valStr + " "*(varStrLen-valStrLen)
            numStr = numStr.replace(varStr,valStr)
        else:
            newVarStr = varStr + " "*(valStrLen-varStrLen)
            numStr = numStr.replace(varStr,valStr)
            symStr = symStr.replace(varStr,newVarStr)

    return MergeableModelFilter(symStr),numStr
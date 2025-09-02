import pyomo.environ as pyo

from .VarToDF import VarToDF


def ModelToDF(model: pyo.ConcreteModel):
    """
    A function that converts a pyomo model containing a result to a dict mapping the name of each variable (str) to a pandas dataframe containing the values of that variable (if the variable is indexed) or to a float containing that variable's value (if the variable is not indexed).

    Parameters
    ----------
    model: pyo.ConcreteModel
        The pyomo model containing the results you'd like to convert.

    Returns
    -------
    dict:
        A dict mapping each variable name (str) to that variable's value. See description for more details.
    """
    vars = {}

    for var in model.component_objects(ctype=pyo.Var):
        varName = str(var)

        readableVarName = varName

        df = VarToDF(var)

        vars[readableVarName] = df

    return vars

import pyomo.environ as pyo
from typing import Union


def DoubleSidedBigM(
    model: pyo.ConcreteModel,
    A: Union[pyo.Var, pyo.Expression],
    B: Union[pyo.Var, pyo.Expression],
    Bmin: Union[float, dict],
    Bmax: Union[float, dict],
    C: Union[pyo.Var, pyo.Expression, float, dict] = 0.0,
    X: Union[pyo.Var, pyo.Expression] = None,
    itrSet: pyo.Set = None,
    includeUpperBounds: bool = True,
    includeLowerBounds: bool = True,
    relationshipBaseName: str = None,
):
    """
    A function to model the following relationship in MILP form:

        A = X * B + C

    where
    * A is a Real number
    * B is a Real number
    * C is a Real number, binary, or parameter
    * X is a binary.

    Parameters
    ----------
    model: pyo.ConcreteModel
        The Pyomo model you'd like to instantiate this relationship within
    A: pyo.Var | pyo.Expression
        The Pyomo variable or expression representing "A" in this relationship
    B: pyo.Var | pyo.Expression
        The Pyomo variable or expression representing "B" in this relationship
    Bmin: float | dict
        A float (if no itrSet is provided) or a dict mapping the elements of itrSet to floats (if itrSet is provided) indicating the minimum possible value of "B"
    Bmax: float | dict
        A float (if no itrSet is provided) or a dict mapping the elements of itrSet to floats (if itrSet is provided) indicating the maximum possible value of "B"
    C: pyo.Var | pyo.Expression | float | dict (optional, Default=0.0)
        The value of "C" in this relationship.
    X: pyo.Var | pyo.Expression (optional, Default = None)
        The Pyomo variable or expression representing "X" in this relationship. Note that if "X" is an expression it must evaluate to a binary value in the true feasible space. If None is provided, a unique Binary variable will be generated
    itrSet: pyo.Set (optional, Default=None)
        The set over which to instantiate this relationship. Note that, if provided, A, B, Bmin, Bmax, C, and X must all be defined over this set. If None is provided, this relationship will be instantiated only for the non-indexed instance.
    includeUpperBounds: bool (optional, Default=True)
        An indication of whether or not you'd like to instantiate the upper bounds of this relationship. Only mark this as False if you're certain that "A" will never be maximized.
    includeLowerBounds: bool (optional, Default=True)
        An indication of whether or not you'd like to instantiate the lower bounds of this relationship. Only mark this as False if you're certain that "A" will never be minimized.
    relationshipBaseName: str (optional, Default=None)
        The base name of the generated constraints and variables for this relationship. If None is provided, one will be generated.

    Returns
    -------
    tuple:
        lowerBound0: pyo.Constraint | None
            The pyomo constraint representing the lower bound of this relationship if X = 0 (if includeLowerBounds is True)
        lowerBound1: pyo.Constraint | None
            The pyomo constraint representing the lower bound of this relationship if X = 1 (if includeLowerBounds is True)
        upperBound0: pyo.Constraint | None
            The pyomo constraint representing the upper bound of this relationship if X = 0 (if includeUpperBounds is True)
        upperBound1: pyo.Constraint | None
            The pyomo constraint representing the upper bound of this relationship if X = 1 (if includeUpperBounds is True)
        X: pyo.Var | pyo.Expression
            The Pyomo variable expression representing "X" in this relationship.
    """
    if relationshipBaseName is None:
        Aname = str(A)
        Bname = str(B)
        if isinstance(C, float) or isinstance(C, dict):
            Cname = (None,)
            Caddage = ""
        else:
            Cname = str(C)
            Caddage = f"_{Cname}"

        relationshipBaseName = f"{Aname}_{Bname}{Caddage}_DoubleSidedBigM"

    if isinstance(C, float) and itrSet is not None:
        C = {idx: C for idx in itrSet}

    lowerBound0Name = f"{relationshipBaseName}_lowerBound0"
    lowerBound1Name = f"{relationshipBaseName}_lowerBound1"
    upperBound0Name = f"{relationshipBaseName}_upperBound0"
    upperBound1Name = f"{relationshipBaseName}_upperBound1"

    if X is None:
        Xname = f"{relationshipBaseName}_X"
        if itrSet is None:
            setattr(model, Xname, pyo.Var(domain=pyo.Binary))
            X = getattr(model, Xname)
        else:
            setattr(model, Xname, pyo.Var(itrSet, domain=pyo.Binary))
            X = getattr(model, Xname)

    if itrSet is None:
        if includeLowerBounds:
            setattr(model, lowerBound0Name, pyo.Constraint(expr=A >= Bmin * X + C))
            lowerBound0 = getattr(model, lowerBound0Name)

            setattr(
                model, lowerBound1Name, pyo.Constraint(expr=A >= B + Bmax * (X - 1) + C)
            )
            lowerBound1 = getattr(model, lowerBound1Name)
        else:
            lowerBound0 = None
            lowerBound1 = None

        if includeUpperBounds:
            setattr(model, upperBound0Name, pyo.Constraint(expr=A <= Bmax * X + C))
            upperBound0 = getattr(model, upperBound0Name)

            setattr(
                model, upperBound1Name, pyo.Constraint(expr=A <= B + Bmin * (X - 1) + C)
            )
            upperBound1 = getattr(model, upperBound1Name)
        else:
            upperBound0 = None
            upperBound1 = None
    else:
        if includeLowerBounds:

            def lowerBound0Func(model, *idx):
                if len(idx) == 1:
                    idx = idx[0]
                return A[idx] >= Bmin[idx] * X[idx] + C[idx]

            setattr(
                model, lowerBound0Name, pyo.Constraint(itrSet, rule=lowerBound0Func)
            )
            lowerBound0 = getattr(model, lowerBound0Name)

            def lowerBound1Func(model, *idx):
                if len(idx) == 1:
                    idx = idx[0]
                return A[idx] >= B[idx] + Bmax[idx] * (X[idx] - 1) + C[idx]

            setattr(
                model, lowerBound1Name, pyo.Constraint(itrSet, rule=lowerBound1Func)
            )
            lowerBound1 = getattr(model, lowerBound1Name)
        else:
            lowerBound0 = None
            lowerBound1 = None

        if includeUpperBounds:

            def upperBound0Func(model, *idx):
                if len(idx) == 1:
                    idx = idx[0]
                return A[idx] <= Bmax[idx] * X[idx] + C[idx]

            setattr(
                model, upperBound0Name, pyo.Constraint(itrSet, rule=upperBound0Func)
            )
            upperBound0 = getattr(model, upperBound0Name)

            def upperBound1Func(model, *idx):
                if len(idx) == 1:
                    idx = idx[0]
                return A[idx] <= B[idx] + Bmin[idx] * (X[idx] - 1) + C[idx]

            setattr(
                model, upperBound1Name, pyo.Constraint(itrSet, rule=upperBound1Func)
            )
            upperBound1 = getattr(model, upperBound1Name)
        else:
            upperBound0 = None
            upperBound1 = None

    return (lowerBound0, lowerBound1, upperBound0, upperBound1, X)

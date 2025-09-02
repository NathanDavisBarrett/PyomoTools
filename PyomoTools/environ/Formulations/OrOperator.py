import pyomo.environ as pyo
from typing import Union


def OrOperator(
    model: pyo.ConcreteModel,
    A: Union[pyo.Var, pyo.Expression],
    B: Union[pyo.Var, pyo.Expression],
    C: Union[pyo.Var, pyo.Expression],
    itrSet: pyo.Set = None,
    relationshipBaseName: str = None,
):
    """
    A function to model the following relationship in MILP or LP form:

        A = B || C

    This is accomplished by the following constraints:

        A <= B + C
        A >= B
        A >= C

    Parameters
    ----------
    model: pyo.ConcreteModel
        The Pyomo model you'd like to instantiate this relationship within
    A: pyo.Var | pyo.Expression
        The Pyomo variable or expression representing "A" in this relationship. Note that "A" should either be or evaluate to a binary value (0 or 1).
    B: pyo.Var | pyo.Expression
        The Pyomo variable or expression representing "B" in this relationship. Note that "B" should either be or evaluate to a binary value (0 or 1).
    C: pyo.Var | pyo.Expression
        The Pyomo variable or expression representing "C" in this relationship. Note that "C" should either be or evaluate to a binary value (0 or 1).
    itrSet: pyo.Set (optional, Default=None)
        The set over which to instantiate this relationship. Note that, if provided, A, B, and C must all be defined over this set. If None is provided, this relationship will be instantiated only for the non-indexed instance.
    relationshipBaseName: str (optional, Default=None)
        The base name of the generated constraints for this relationship. If None is provided, one will be generated.

    Returns
    -------
    tuple:
        constraint1: pyo.Constraint
            The pyomo constraint representing A <= B + C
        constraint2: pyo.Constraint
            The pyomo constraint representing A >= B
        constraint3: pyo.Constraint
            The pyomo constraint representing A >= C
    """
    if relationshipBaseName is None:
        Aname = str(A)
        Bname = str(B)
        Cname = str(C)
        relationshipBaseName = f"{Aname}_{Bname}_{Cname}_OrOperator"

    constraint1Name = f"{relationshipBaseName}_constraint1"
    constraint2Name = f"{relationshipBaseName}_constraint2"
    constraint3Name = f"{relationshipBaseName}_constraint3"

    if itrSet is None:
        setattr(model, constraint1Name, pyo.Constraint(expr=A <= B + C))
        constraint1 = getattr(model, constraint1Name)

        setattr(model, constraint2Name, pyo.Constraint(expr=A >= B))
        constraint2 = getattr(model, constraint2Name)

        setattr(model, constraint3Name, pyo.Constraint(expr=A >= C))
        constraint3 = getattr(model, constraint3Name)
    else:

        def constraint1Func(model, *idx):
            if len(idx) == 1:
                idx = idx[0]
            return A[idx] <= B[idx] + C[idx]

        setattr(model, constraint1Name, pyo.Constraint(itrSet, rule=constraint1Func))
        constraint1 = getattr(model, constraint1Name)

        def constraint2Func(model, *idx):
            if len(idx) == 1:
                idx = idx[0]
            return A[idx] >= B[idx]

        setattr(model, constraint2Name, pyo.Constraint(itrSet, rule=constraint2Func))
        constraint2 = getattr(model, constraint2Name)

        def constraint3Func(model, *idx):
            if len(idx) == 1:
                idx = idx[0]
            return A[idx] >= C[idx]

        setattr(model, constraint3Name, pyo.Constraint(itrSet, rule=constraint3Func))
        constraint3 = getattr(model, constraint3Name)

    return (constraint1, constraint2, constraint3)

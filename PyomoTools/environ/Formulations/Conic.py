import pyomo.environ as pyo
from typing import Iterable, Union


def Conic(
    model: pyo.ConcreteModel,
    x: Iterable[Union[pyo.Var, pyo.Expression, float]],
    r: Union[pyo.Var, pyo.Expression, float],
    order: int,
    constraintBaseName: str,
    explicit_r_nonnegativity: bool = True,
):
    """
    Function for enforcing conic constraints in pyomo.environ.

    Parameters
    ----------
    model : pyo.ConcreteModel
        The Pyomo model to add constraints to.
    x : Iterable[Union[pyo.Var, pyo.Expression, float]]
        The left-hand side of the conic constraint.
    r : Union[pyo.Var, pyo.Expression, float]
        The right-hand side of the conic constraint.
    order : int
        The order of the conic constraint (must be positive integer).
    constraintBaseName : str
        Base name for generated constraints.
    explicit_r_nonnegativity : bool, optional
        Whether to explicitly enforce the non-negativity of r. Default is True.


    Returns
    -------
    tuple
        (conic_constraint, r_nonnegativity_constraint, conic_constraint_neg)
        Only returns constraints that are created (None if not applicable).
    """
    assert isinstance(order, int) and order > 0, "Order must be a positive integer."

    r_nonnegativity = None
    if explicit_r_nonnegativity:
        cname = f"{constraintBaseName}_r_nonnegativity"
        setattr(model, cname, pyo.Constraint(expr=r >= 0))
        r_nonnegativity = getattr(model, cname)

    cname = f"{constraintBaseName}_conic"
    setattr(model, cname, pyo.Constraint(expr=sum(xi**order for xi in x) <= r**order))
    conic_constraint = getattr(model, cname)

    conic_constraint_neg = None
    if order % 2 != 0:
        cname = f"{constraintBaseName}_conic_neg"
        setattr(
            model, cname, pyo.Constraint(expr=sum(-(xi**order) for xi in x) <= r**order)
        )
        conic_constraint_neg = getattr(model, cname)

    return (conic_constraint, r_nonnegativity, conic_constraint_neg)

import pyomo.environ as pyo
from typing import Iterable, Union

from .ConditionalGeq import ConditionalGeq
from .ConditionalLeq import ConditionalLeq


def ConstrainContiguousBinarySeries(
    model: pyo.ConcreteModel,
    X: Iterable[Union[pyo.Var, pyo.Expression]],
    minimumLength: int = 1,
    maximumLength: int = None,
    relationshipBaseName: str = None,
):
    """
    A function to model the following relationship in MILP form:

        Given X[t], a series of binary variables, only a contiguous sub-series of them can be 1-valued.

    Examples where this behavior arises include:
    * A series of binary variables representing whether or not a machine is on at each time step t. The machine can only turn on and off one time in the series.

    A very tight (though not necessarily compact) formulation of this relationship involves an Integer start and end index, StartIndex and EndIndex, such that X[i] = 1 if and only if StartIndex <= i <= EndIndex.

    Parameters
    ----------
    model: pyo.ConcreteModel
        The Pyomo model you'd like to instantiate this relationship within
    X: Iterable[pyo.Var | pyo.Expression]
        An iterable of Pyomo binary variables representing the series of binary variables to constrain.
    minimumLength: int (optional, Default=1)
        The minimum length of the contiguous sub-series that can be 1-valued.
    maximumLength: int (optional, Default=None)
        The maximum length of the contiguous sub-series that can be 1-valued. If None, there is no maximum length constraint.
    relationshipBaseName: str (optional, Default=None)
        The base name of the generated constraints and variables for this relationship. If None is provided, one will be generated.

    Returns
    -------
    tuple:
        StartIndex: pyo.Var
            The integer variable representing the start index of the contiguous series
        EndIndex: pyo.Var
            The integer variable representing the end index of the contiguous series
        constraints: dict
            A dictionary containing all the generated constraints and variables
    """
    if relationshipBaseName is None:
        relationshipBaseName = "ConstrainContiguousBinarySeries"

    X = list(X)
    n = len(X)

    minIndex = 0
    maxIndex = n - 1

    # Create StartIndex and EndIndex variables
    StartIndexName = f"{relationshipBaseName}_StartIndex"
    EndIndexName = f"{relationshipBaseName}_EndIndex"

    setattr(
        model,
        StartIndexName,
        pyo.Var(domain=pyo.Integers, bounds=(minIndex, maxIndex), initialize=minIndex),
    )
    StartIndex = getattr(model, StartIndexName)

    setattr(
        model,
        EndIndexName,
        pyo.Var(domain=pyo.Integers, bounds=(minIndex, maxIndex), initialize=maxIndex),
    )
    EndIndex = getattr(model, EndIndexName)

    # Minimum length constraint
    minimumLengthName = f"{relationshipBaseName}_MinimumLength"
    setattr(
        model,
        minimumLengthName,
        pyo.Constraint(expr=StartIndex <= EndIndex - minimumLength + 1),
    )
    minimumLengthConstraint = getattr(model, minimumLengthName)

    # Maximum length constraint (if specified)
    maximumLengthConstraint = None
    if maximumLength is not None:
        maximumLengthName = f"{relationshipBaseName}_MaximumLength"
        setattr(
            model,
            maximumLengthName,
            pyo.Constraint(expr=EndIndex <= StartIndex + maximumLength - 1),
        )
        maximumLengthConstraint = getattr(model, maximumLengthName)

    # Create IsAfterStart and IsBeforeEnd variables
    IsAfterStartName = f"{relationshipBaseName}_IsAfterStart"
    IsBeforeEndName = f"{relationshipBaseName}_IsBeforeEnd"

    setattr(model, IsAfterStartName, pyo.Var(range(n), domain=pyo.Binary, initialize=0))
    IsAfterStart = getattr(model, IsAfterStartName)

    setattr(model, IsBeforeEndName, pyo.Var(range(n), domain=pyo.Binary, initialize=0))
    IsBeforeEnd = getattr(model, IsBeforeEndName)

    # AfterStart definition: IsAfterStart[i] = 1 if StartIndex <= i
    afterStartConstraints = {}
    for i in range(n):
        afterStartName = f"{relationshipBaseName}_AfterStart_{i}"
        # Use ConditionalLeq: StartIndex <= i if IsAfterStart[i] == 1
        upperBound, lowerBound, _ = ConditionalLeq(
            model=model,
            A=StartIndex,
            alpha=i,
            X=IsAfterStart[i],
            epsilon=1,  # 1 since StartIndex is integer
            A_bounds=(minIndex, maxIndex),
            relationshipBaseName=afterStartName,
        )
        afterStartConstraints[i] = (upperBound, lowerBound)

    # BeforeEnd definition: IsBeforeEnd[i] = 1 if EndIndex >= i
    beforeEndConstraints = {}
    for i in range(n):
        beforeEndName = f"{relationshipBaseName}_BeforeEnd_{i}"
        # Use ConditionalGeq: EndIndex >= i if IsBeforeEnd[i] == 1
        upperBound, lowerBound, _ = ConditionalGeq(
            model=model,
            A=EndIndex,
            alpha=i,
            X=IsBeforeEnd[i],
            epsilon=1,  # 1 since EndIndex is integer
            A_bounds=(minIndex, maxIndex),
            relationshipBaseName=beforeEndName,
        )
        beforeEndConstraints[i] = (upperBound, lowerBound)

    # X[i] is 1 if it is both after the start and before the end
    XDefinitionName = f"{relationshipBaseName}_XDefinition"

    def xDefinitionFunc(model, i):
        return X[i] == IsAfterStart[i] + IsBeforeEnd[i] - 1

    setattr(model, XDefinitionName, pyo.Constraint(range(n), rule=xDefinitionFunc))
    XDefinition = getattr(model, XDefinitionName)

    constraints = {
        "MinimumLength": minimumLengthConstraint,
        "MaximumLength": maximumLengthConstraint,
        "AfterStart": afterStartConstraints,
        "BeforeEnd": beforeEndConstraints,
        "XDefinition": XDefinition,
        "IsAfterStart": IsAfterStart,
        "IsBeforeEnd": IsBeforeEnd,
    }

    return (StartIndex, EndIndex, constraints)

from ...base.Formulations.PWL1D import PWL1DParameters, PWL1DType

import pyomo.environ as pyo
from typing import Union, Tuple, Dict, Any


def PWL1D(
    model: pyo.ConcreteModel,
    params: Union[PWL1DParameters, Dict[Any, PWL1DParameters]],
    xVar: Union[pyo.Var, pyo.Expression],
    yVar: Union[pyo.Var, pyo.Expression],
    itrSet: pyo.Set = None,
    relationshipBaseName: str = None,
):
    """
    A function to model a piecewise linear function in MILP or LP form.

    The function creates the appropriate constraints based on the PWL type:
    - LINEAR: Single linear equality constraint
    - CONVEX: Linear inequalities for convex hull
    - CONCAVE: Linear inequalities for concave envelope
    - GENERAL: SOS2 formulation with weights

    Parameters
    ----------
    model: pyo.ConcreteModel
        The Pyomo model you'd like to instantiate this relationship within
    params: PWL1DParameters | Dict[Any, PWL1DParameters]
        The parameters defining the piecewise linear function. If itrSet is None,
        this should be a single PWL1DParameters object. If itrSet is provided,
        this should be a dictionary mapping each index in itrSet to its own
        PWL1DParameters object.
    xVar: pyo.Var | pyo.Expression
        The Pyomo variable or expression representing the x-coordinate
    yVar: pyo.Var | pyo.Expression
        The Pyomo variable or expression representing the y-coordinate
    itrSet: pyo.Set (optional, Default=None)
        The set over which to instantiate this relationship. Note that, if provided,
        xVar and yVar must all be defined over this set, and params must be a
        dictionary mapping each index to its PWL1DParameters. If None is provided,
        this relationship will be instantiated only for the non-indexed instance.
    relationshipBaseName: str (optional, Default=None)
        The base name of the generated constraints and variables for this relationship.
        If None is provided, one will be generated.

    Returns
    -------
    tuple:
        The returned tuple varies based on the PWL type:
        - LINEAR: (linearEquality,)
        - CONVEX: (convexInequalities,)
        - CONCAVE: (concaveInequalities,)
        - GENERAL: (weights, weightSumConstraint, sos2Constraint, xValueConstraint, yValueConstraint)
    """
    if relationshipBaseName is None:
        xname = str(xVar)
        yname = str(yVar)
        relationshipBaseName = f"{xname}_{yname}_PWL1D"

    # Handle both single params and dictionary of params
    if itrSet is None:
        # Non-indexed case: params should be a single PWL1DParameters object
        if isinstance(params, dict):
            raise ValueError(
                "When itrSet is None, params should be a single PWL1DParameters object, not a dictionary"
            )

        pwl_type = params.pwl_type

        if pwl_type == PWL1DType.LINEAR:
            return _init_linear(model, params, xVar, yVar, relationshipBaseName)
        elif pwl_type == PWL1DType.CONVEX:
            return _init_convex(model, params, xVar, yVar, relationshipBaseName)
        elif pwl_type == PWL1DType.CONCAVE:
            return _init_concave(model, params, xVar, yVar, relationshipBaseName)
        elif pwl_type == PWL1DType.GENERAL:
            return _init_general(model, params, xVar, yVar, relationshipBaseName)
        else:
            raise ValueError(f"Unsupported PWL type: {pwl_type}")
    else:
        # Indexed case: params should be a dictionary
        if not isinstance(params, dict):
            raise ValueError(
                "When itrSet is provided, params should be a dictionary mapping each index to PWL1DParameters"
            )

        # Check that all indices in itrSet have corresponding params
        for idx in itrSet:
            if idx not in params:
                raise ValueError(
                    f"Missing PWL1DParameters for index {idx} in params dictionary"
                )

        results = {}

        for idx in itrSet:
            params_idx = params[idx]
            pwl_type = params_idx.pwl_type
            idxName = f"{relationshipBaseName}_{idx}"
            if pwl_type == PWL1DType.LINEAR:
                results[idx] = _init_linear(
                    model, params_idx, xVar[idx], yVar[idx], idxName
                )
            elif pwl_type == PWL1DType.CONVEX:
                results[idx] = _init_convex(
                    model, params_idx, xVar[idx], yVar[idx], idxName
                )
            elif pwl_type == PWL1DType.CONCAVE:
                results[idx] = _init_concave(
                    model, params_idx, xVar[idx], yVar[idx], idxName
                )
            elif pwl_type == PWL1DType.GENERAL:
                results[idx] = _init_general(
                    model, params_idx, xVar[idx], yVar[idx], idxName
                )
            else:
                raise ValueError(f"Unsupported PWL type: {pwl_type}")
        return results


def _init_linear(
    model: pyo.ConcreteModel,
    params: PWL1DParameters,
    xVar: Union[pyo.Var, pyo.Expression],
    yVar: Union[pyo.Var, pyo.Expression],
    relationshipBaseName: str,
) -> Tuple[pyo.Constraint]:
    """Initialize linear PWL formulation."""
    constraintName = f"{relationshipBaseName}_linearEquality"

    p1, p2 = params.points[0], params.points[1]
    setattr(
        model,
        constraintName,
        pyo.Constraint(
            expr=(yVar - p1[1]) * (p2[0] - p1[0]) == (xVar - p1[0]) * (p2[1] - p1[1])
        ),
    )
    constraint = getattr(model, constraintName)

    _enforce_x_bounds(xVar, params)

    return (constraint,)


def _enforce_x_bounds(xVar: pyo.Var, params: PWL1DParameters):
    if params.includeLB_x:
        existingLB = xVar.lb
        if existingLB is None or existingLB < params.points[0][0]:
            xVar.lb = params.points[0][0]

    if params.includeUB_x:
        existingUB = xVar.ub
        if existingUB is None or existingUB > params.points[-1][0]:
            xVar.ub = params.points[-1][0]


def _init_convex(
    model: pyo.ConcreteModel,
    params: PWL1DParameters,
    xVar: Union[pyo.Var, pyo.Expression],
    yVar: Union[pyo.Var, pyo.Expression],
    relationshipBaseName: str,
) -> Tuple[pyo.Constraint]:
    """Initialize convex PWL formulation."""
    constraintName = f"{relationshipBaseName}_convexInequalities"

    def constraintFunc(model, i):
        p1, p2 = params.points[i - 1], params.points[i]
        dx = p2[0] - p1[0]
        if dx >= 0:
            return (yVar - p1[1]) * dx >= (xVar - p1[0]) * (p2[1] - p1[1])
        else:
            return (yVar - p1[1]) * dx <= (xVar - p1[0]) * (p2[1] - p1[1])

    segmentSet = pyo.RangeSet(1, len(params.points) - 1)
    setattr(model, f"{relationshipBaseName}_segments", segmentSet)
    setattr(model, constraintName, pyo.Constraint(segmentSet, rule=constraintFunc))
    constraint = getattr(model, constraintName)

    _enforce_x_bounds(xVar, params)

    return (constraint,)


def _init_concave(
    model: pyo.ConcreteModel,
    params: PWL1DParameters,
    xVar: Union[pyo.Var, pyo.Expression],
    yVar: Union[pyo.Var, pyo.Expression],
    relationshipBaseName: str,
) -> Tuple[pyo.Constraint]:
    """Initialize concave PWL formulation."""
    constraintName = f"{relationshipBaseName}_concaveInequalities"

    def constraintFunc(model, i):
        p1, p2 = params.points[i - 1], params.points[i]
        dx = p2[0] - p1[0]
        if dx >= 0:
            return (yVar - p1[1]) * dx <= (xVar - p1[0]) * (p2[1] - p1[1])
        else:
            return (yVar - p1[1]) * dx >= (xVar - p1[0]) * (p2[1] - p1[1])

    segmentSet = pyo.RangeSet(1, len(params.points) - 1)
    setattr(model, f"{relationshipBaseName}_segments", segmentSet)
    setattr(model, constraintName, pyo.Constraint(segmentSet, rule=constraintFunc))
    constraint = getattr(model, constraintName)

    _enforce_x_bounds(xVar, params)

    return (constraint,)


def _init_general(
    model: pyo.ConcreteModel,
    params: Union[PWL1DParameters, Dict[Any, PWL1DParameters]],
    xVar: Union[pyo.Var, pyo.Expression],
    yVar: Union[pyo.Var, pyo.Expression],
    relationshipBaseName: str,
) -> Tuple[pyo.Var, pyo.Constraint, pyo.SOSConstraint, pyo.Constraint, pyo.Constraint]:
    """Initialize general PWL formulation using SOS2."""
    weightsName = f"{relationshipBaseName}_weights"
    weightSumConstraintName = f"{relationshipBaseName}_weightSum"
    sos2ConstraintName = f"{relationshipBaseName}_sos2"
    xValueConstraintName = f"{relationshipBaseName}_xValue"
    yValueConstraintName = f"{relationshipBaseName}_yValue"

    pointSet = pyo.RangeSet(0, params.num_points - 1)
    setattr(model, f"{relationshipBaseName}_points", pointSet)
    setattr(model, weightsName, pyo.Var(pointSet, domain=pyo.NonNegativeReals))
    weights = getattr(model, weightsName)

    # Weight sum constraint
    setattr(
        model,
        weightSumConstraintName,
        pyo.Constraint(expr=sum(weights[i] for i in pointSet) == 1),
    )
    weightSumConstraint = getattr(model, weightSumConstraintName)

    # SOS2 constraint
    setattr(model, sos2ConstraintName, pyo.SOSConstraint(var=weights, sos=2))
    sos2Constraint = getattr(model, sos2ConstraintName)

    # X value constraint
    setattr(
        model,
        xValueConstraintName,
        pyo.Constraint(
            expr=xVar == sum(weights[i] * params.points[i][0] for i in pointSet)
        ),
    )
    xValueConstraint = getattr(model, xValueConstraintName)

    # Y value constraint
    setattr(
        model,
        yValueConstraintName,
        pyo.Constraint(
            expr=yVar == sum(weights[i] * params.points[i][1] for i in pointSet)
        ),
    )
    yValueConstraint = getattr(model, yValueConstraintName)

    _enforce_x_bounds(xVar, params)

    return (
        weights,
        weightSumConstraint,
        sos2Constraint,
        xValueConstraint,
        yValueConstraint,
    )

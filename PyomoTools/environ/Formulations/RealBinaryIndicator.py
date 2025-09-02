import pyomo.environ as pyo
import numpy as np
import enum
from typing import Union
from warnings import warn


class Relation(enum.Enum):
    GEQ = 0
    LEQ = 1
    EQ = 2


def RealBinaryIndicator(
    model: pyo.ConcreteModel,
    X: Union[pyo.Var, pyo.Expression],
    Y: Union[pyo.Var, pyo.Expression],
    A: Union[pyo.Var, pyo.Expression],
    alphaPrime: float,
    alphaMin: float,
    alphaMax: float,
    aRelation: Relation = Relation.GEQ,
    epsilon: float = 1e-6,
    itrSet: pyo.Set = None,
    relationshipBaseName: str = None,
):
    """
    A function to model the following relationship in MILP form:

        X = 1 if (A >= alphaPrime) and (Y = 1) else 0

    or

        X = 1 if (A <= alphaPrime) and (Y = 1) else 0

    (depending of the Relation provided)

    where
    * A is a continuous variable (or expression) between alphaMin and alphaMax
    * alphaMin <= alphaPrime <= alphaMax are constant parameters
    * Y is a binary variable (or expression that evaluates to binary)
    * X is a binary variable (or expression that evaluates to binary)

    This is accomplished by constraining three planes defined by the following collections of (X,Y,A) points:

    Relation.GEQ:
        1. (0,0,alphaMin), (0,1,alphaMin), (1,1,alphaPrime)
        2. (0,0,alphaMin), (1,1,alphaMin), (0,0,alphaMax)
        3. (0,1,alphaPrime-epsilon), (0,0,alphaMax), (1,1,alphaMax)

    Relation.LEQ:
        1. (0,0,alphaMin), (0,1,alphaPrime+epsilon), (1,1,alphaMin)
        2. (0,0,alphaMin), (1,1,alphaMin), (0,0,alphaMax)
        3. (0,1,alphaMax), (1,1,alphaPrime), (0,0,alphaMax)

    Parameters
    ----------
    model: pyo.ConcreteModel
        The Pyomo model you'd like to instantiate this relationship within
    X: pyo.Var | pyo.Expression
        The Pyomo variable or expression representing "X" in this relationship. Should be binary.
    Y: pyo.Var | pyo.Expression
        The Pyomo variable or expression representing "Y" in this relationship. Should be binary.
    A: pyo.Var | pyo.Expression
        The Pyomo variable or expression representing "A" in this relationship. Should be continuous.
    alphaPrime: float | dict
        The threshold value for the relationship. Can be a dict if itrSet is provided.
    alphaMin: float | dict
        The minimum possible value of "A". Can be a dict if itrSet is provided.
    alphaMax: float | dict
        The maximum possible value of "A". Can be a dict if itrSet is provided.
    aRelation: Relation (optional, Default=Relation.GEQ)
        Whether the relationship is >= (GEQ) or <= (LEQ).
    epsilon: float | dict (optional, Default=1e-6)
        A small positive constant for numerical stability. Can be a dict if itrSet is provided.
    itrSet: pyo.Set (optional, Default=None)
        The set over which to instantiate this relationship. If provided, all variables must be indexed over this set.
    relationshipBaseName: str (optional, Default=None)
        The base name of the generated constraints for this relationship. If None is provided, one will be generated.

    Returns
    -------
    list:
        constraints: list of pyo.Constraint
            The pyomo constraints implementing this relationship
    """
    if relationshipBaseName is None:
        Xname = str(X)
        Yname = str(Y)
        Aname = str(A)
        relationshipBaseName = f"{Xname}_{Yname}_{Aname}_RealBinaryIndicator"

    # Convert scalars to dicts if itrSet is provided
    if itrSet is not None:
        if isinstance(alphaPrime, (int, float)):
            alphaPrime = {idx: alphaPrime for idx in itrSet}
        if isinstance(alphaMin, (int, float)):
            alphaMin = {idx: alphaMin for idx in itrSet}
        if isinstance(alphaMax, (int, float)):
            alphaMax = {idx: alphaMax for idx in itrSet}
        if isinstance(epsilon, (int, float)):
            epsilon = {idx: epsilon for idx in itrSet}

    def computePlane(points):
        """
        Computes the plane defined by three points in 3D space.
        Returns coefficients [C1, C2, C3, C4] for: C1*X + C2*Y + C3*A = C4
        """
        points = np.asarray(points)
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        normal = np.cross(v1, v2)
        d = np.dot(normal, points[0])
        return np.array([normal[0], normal[1], normal[2], d])

    def createConstraint(
        model, constraintName, points, violatingPoint, X_val, Y_val, A_val, idx=None
    ):
        """Create a constraint based on three points and a violating point."""
        C = computePlane(points)

        # Check which side of the plane the violating point is on
        predictedValue = C[:3] @ np.array(violatingPoint)

        if idx is None:
            # Non-indexed case
            if predictedValue < C[3]:
                # Use >= constraint
                setattr(
                    model,
                    constraintName,
                    pyo.Constraint(
                        expr=C[0] * X_val + C[1] * Y_val + C[2] * A_val >= C[3]
                    ),
                )
            else:
                # Use <= constraint
                setattr(
                    model,
                    constraintName,
                    pyo.Constraint(
                        expr=C[0] * X_val + C[1] * Y_val + C[2] * A_val <= C[3]
                    ),
                )
        else:
            # Indexed case
            def constraintFunc(model, *idx_args):
                if len(idx_args) == 1:
                    idx_args = idx_args[0]
                if predictedValue < C[3]:
                    return (
                        C[0] * X_val[idx_args]
                        + C[1] * Y_val[idx_args]
                        + C[2] * A_val[idx_args]
                        >= C[3]
                    )
                else:
                    return (
                        C[0] * X_val[idx_args]
                        + C[1] * Y_val[idx_args]
                        + C[2] * A_val[idx_args]
                        <= C[3]
                    )

            setattr(model, constraintName, pyo.Constraint(itrSet, rule=constraintFunc))

        return getattr(model, constraintName)

    constraints = []

    if itrSet is None:
        # Non-indexed case
        if epsilon <= 0:
            warn(
                f"Epsilon value {epsilon} is non-positive. This may lead to numerical issues in the formulation.",
                UserWarning,
            )

        assert (
            alphaMin <= alphaMax
        ), f"alphaMin ({alphaMin}) must be less than or equal to alphaMax ({alphaMax})"

        if aRelation == Relation.GEQ:
            # X = 1 if (A >= alphaPrime) and (Y = 1) else 0
            if alphaPrime > alphaMax:
                # X = 0 always
                constraintName = f"{relationshipBaseName}_X_zero"
                setattr(model, constraintName, pyo.Constraint(expr=X == 0))
                constraints.append(getattr(model, constraintName))
            elif alphaPrime < alphaMin:
                # X = Y always
                constraintName = f"{relationshipBaseName}_X_equals_Y"
                setattr(model, constraintName, pyo.Constraint(expr=X == Y))
                constraints.append(getattr(model, constraintName))
            else:
                # Three planar constraints
                constraintName1 = f"{relationshipBaseName}_plane1"
                constraint1 = createConstraint(
                    model,
                    constraintName1,
                    [(0, 0, alphaMin), (0, 1, alphaMin), (1, 1, alphaPrime)],
                    (1, 1, alphaMin),
                    X,
                    Y,
                    A,
                )
                constraints.append(constraint1)

                constraintName2 = f"{relationshipBaseName}_plane2"
                constraint2 = createConstraint(
                    model,
                    constraintName2,
                    [(0, 0, alphaMin), (1, 1, alphaMin), (0, 0, alphaMax)],
                    (1, 0, alphaMin),
                    X,
                    Y,
                    A,
                )
                constraints.append(constraint2)

                constraintName3 = f"{relationshipBaseName}_plane3"
                constraint3 = createConstraint(
                    model,
                    constraintName3,
                    [(0, 1, alphaPrime - epsilon), (0, 0, alphaMax), (1, 1, alphaMax)],
                    (0, 1, alphaMax),
                    X,
                    Y,
                    A,
                )
                constraints.append(constraint3)

        elif aRelation == Relation.LEQ:
            # X = 1 if (A <= alphaPrime) and (Y = 1) else 0
            if alphaPrime > alphaMax:
                # X = Y always
                constraintName = f"{relationshipBaseName}_X_equals_Y"
                setattr(model, constraintName, pyo.Constraint(expr=X == Y))
                constraints.append(getattr(model, constraintName))
            elif alphaPrime < alphaMin:
                # X = 0 always
                constraintName = f"{relationshipBaseName}_X_zero"
                setattr(model, constraintName, pyo.Constraint(expr=X == 0))
                constraints.append(getattr(model, constraintName))
            else:
                # Three planar constraints
                constraintName1 = f"{relationshipBaseName}_plane1"
                constraint1 = createConstraint(
                    model,
                    constraintName1,
                    [(0, 0, alphaMin), (0, 1, alphaPrime + epsilon), (1, 1, alphaMin)],
                    (0, 1, alphaMin),
                    X,
                    Y,
                    A,
                )
                constraints.append(constraint1)

                constraintName2 = f"{relationshipBaseName}_plane2"
                constraint2 = createConstraint(
                    model,
                    constraintName2,
                    [(0, 0, alphaMin), (1, 1, alphaMin), (0, 0, alphaMax)],
                    (1, 0, alphaMin),
                    X,
                    Y,
                    A,
                )
                constraints.append(constraint2)

                constraintName3 = f"{relationshipBaseName}_plane3"
                constraint3 = createConstraint(
                    model,
                    constraintName3,
                    [(0, 1, alphaMax), (1, 1, alphaPrime), (0, 0, alphaMax)],
                    (1, 1, alphaMax),
                    X,
                    Y,
                    A,
                )
                constraints.append(constraint3)

    else:
        # Indexed case - create constraints for each index
        for idx in itrSet:
            eps_val = epsilon[idx] if isinstance(epsilon, dict) else epsilon
            alpha_prime_val = (
                alphaPrime[idx] if isinstance(alphaPrime, dict) else alphaPrime
            )
            alpha_min_val = alphaMin[idx] if isinstance(alphaMin, dict) else alphaMin
            alpha_max_val = alphaMax[idx] if isinstance(alphaMax, dict) else alphaMax

            if eps_val <= 0:
                warn(
                    f"Epsilon value {eps_val} for index {idx} is non-positive.",
                    UserWarning,
                )

            assert (
                alpha_min_val <= alpha_max_val
            ), f"alphaMin ({alpha_min_val}) must be <= alphaMax ({alpha_max_val}) for index {idx}"

            if aRelation == Relation.GEQ:
                if alpha_prime_val > alpha_max_val:
                    # X[idx] = 0 always
                    constraintName = f"{relationshipBaseName}_X_zero_{idx}"

                    def xZeroFunc(model, *idx_args):
                        if len(idx_args) == 1:
                            idx_args = idx_args[0]
                        return (
                            X[idx_args] == 0 if idx_args == idx else pyo.Constraint.Skip
                        )

                    setattr(
                        model, constraintName, pyo.Constraint(itrSet, rule=xZeroFunc)
                    )
                    constraints.append(getattr(model, constraintName))
                elif alpha_prime_val < alpha_min_val:
                    # X[idx] = Y[idx] always
                    constraintName = f"{relationshipBaseName}_X_equals_Y_{idx}"

                    def xEqualsYFunc(model, *idx_args):
                        if len(idx_args) == 1:
                            idx_args = idx_args[0]
                        return (
                            X[idx_args] == Y[idx_args]
                            if idx_args == idx
                            else pyo.Constraint.Skip
                        )

                    setattr(
                        model, constraintName, pyo.Constraint(itrSet, rule=xEqualsYFunc)
                    )
                    constraints.append(getattr(model, constraintName))
                else:
                    # Three planar constraints - simplified for indexed case
                    constraintName1 = f"{relationshipBaseName}_plane1_{idx}"
                    constraint1 = createConstraint(
                        model,
                        constraintName1,
                        [
                            (0, 0, alpha_min_val),
                            (0, 1, alpha_min_val),
                            (1, 1, alpha_prime_val),
                        ],
                        (1, 1, alpha_min_val),
                        X,
                        Y,
                        A,
                        idx,
                    )
                    constraints.append(constraint1)

                    constraintName2 = f"{relationshipBaseName}_plane2_{idx}"
                    constraint2 = createConstraint(
                        model,
                        constraintName2,
                        [
                            (0, 0, alpha_min_val),
                            (1, 1, alpha_min_val),
                            (0, 0, alpha_max_val),
                        ],
                        (1, 0, alpha_min_val),
                        X,
                        Y,
                        A,
                        idx,
                    )
                    constraints.append(constraint2)

                    constraintName3 = f"{relationshipBaseName}_plane3_{idx}"
                    constraint3 = createConstraint(
                        model,
                        constraintName3,
                        [
                            (0, 1, alpha_prime_val - eps_val),
                            (0, 0, alpha_max_val),
                            (1, 1, alpha_max_val),
                        ],
                        (0, 1, alpha_max_val),
                        X,
                        Y,
                        A,
                        idx,
                    )
                    constraints.append(constraint3)

            elif aRelation == Relation.LEQ:
                if alpha_prime_val > alpha_max_val:
                    # X[idx] = Y[idx] always
                    constraintName = f"{relationshipBaseName}_X_equals_Y_{idx}"

                    def xEqualsYFunc(model, *idx_args):
                        if len(idx_args) == 1:
                            idx_args = idx_args[0]
                        return (
                            X[idx_args] == Y[idx_args]
                            if idx_args == idx
                            else pyo.Constraint.Skip
                        )

                    setattr(
                        model, constraintName, pyo.Constraint(itrSet, rule=xEqualsYFunc)
                    )
                    constraints.append(getattr(model, constraintName))
                elif alpha_prime_val < alpha_min_val:
                    # X[idx] = 0 always
                    constraintName = f"{relationshipBaseName}_X_zero_{idx}"

                    def xZeroFunc(model, *idx_args):
                        if len(idx_args) == 1:
                            idx_args = idx_args[0]
                        return (
                            X[idx_args] == 0 if idx_args == idx else pyo.Constraint.Skip
                        )

                    setattr(
                        model, constraintName, pyo.Constraint(itrSet, rule=xZeroFunc)
                    )
                    constraints.append(getattr(model, constraintName))
                else:
                    # Three planar constraints
                    constraintName1 = f"{relationshipBaseName}_plane1_{idx}"
                    constraint1 = createConstraint(
                        model,
                        constraintName1,
                        [
                            (0, 0, alpha_min_val),
                            (0, 1, alpha_prime_val + eps_val),
                            (1, 1, alpha_min_val),
                        ],
                        (0, 1, alpha_min_val),
                        X,
                        Y,
                        A,
                        idx,
                    )
                    constraints.append(constraint1)

                    constraintName2 = f"{relationshipBaseName}_plane2_{idx}"
                    constraint2 = createConstraint(
                        model,
                        constraintName2,
                        [
                            (0, 0, alpha_min_val),
                            (1, 1, alpha_min_val),
                            (0, 0, alpha_max_val),
                        ],
                        (1, 0, alpha_min_val),
                        X,
                        Y,
                        A,
                        idx,
                    )
                    constraints.append(constraint2)

                    constraintName3 = f"{relationshipBaseName}_plane3_{idx}"
                    constraint3 = createConstraint(
                        model,
                        constraintName3,
                        [
                            (0, 1, alpha_max_val),
                            (1, 1, alpha_prime_val),
                            (0, 0, alpha_max_val),
                        ],
                        (1, 1, alpha_max_val),
                        X,
                        Y,
                        A,
                        idx,
                    )
                    constraints.append(constraint3)

    return constraints

import pyomo.environ as pyo
from typing import Union, Tuple


def ConditionalLeq(
    model: pyo.ConcreteModel,
    A: Union[pyo.Var, pyo.Expression],
    alpha: float,
    X: Union[pyo.Var, pyo.Expression] = None,
    epsilon: float = 1e-5,
    A_bounds: Tuple[float, float] = None,
    itrSet: pyo.Set = None,
    relationshipBaseName: str = None,
):
    """
    A function to model the following relationship in MILP form:

        A > alpha  if X == 0
        A <= alpha if X == 1

    Or, more precisely:

        A >= alpha + epsilon  if X == 0
        A <= alpha            if X == 1

    where
    * A is a variable (real or integer)
    * X is a binary
    * alpha is a constant parameter
    * epsilon is a small positive constant

    Ideologically, this relationship is the same as ConditionalGeq, but with A' = 2*alpha - A

    Parameters
    ----------
    model: pyo.ConcreteModel
        The Pyomo model you'd like to instantiate this relationship within
    A: pyo.Var | pyo.Expression
        The Pyomo variable or expression representing "A" in this relationship.
    alpha: float | dict
        A float (if no itrSet is provided) or a dict mapping the elements of itrSet to floats (if itrSet is provided) indicating the value of "alpha" in this relationship. This is the threshold that "A" must not exceed when "X" is 1.
    X: pyo.Var | pyo.Expression (optional, Default = None)
        The Pyomo variable or expression representing "X" in this relationship. Note that if "X" is an expression, it must evaluate to a binary value in the true feasible space. If None is provided, a unique Binary variable will be generated.
    epsilon: float | dict (optional, Default=1e-5)
        A small positive constant to ensure strict inequality when "X" is 0. Can be a dict if itrSet is provided.
    A_bounds: Tuple[float,float] | dict (optional, Default=None)
        A tuple indicating the minimum and maximum possible values of "A". This is required if "A" does not have intrinsic bounds already defined or if "A" is an expression. Can be a dict mapping indices to tuples if itrSet is provided.
    itrSet: pyo.Set (optional, Default=None)
        The set over which to instantiate this relationship. Note that, if provided, A, alpha, epsilon, A_bounds, and X must all be defined over this set. If None is provided, this relationship will be instantiated only for the non-indexed instance.
    relationshipBaseName: str (optional, Default=None)
        The base name of the generated constraints and variables for this relationship. If None is provided, one will be generated.

    Returns
    -------
    tuple:
        upperBound: pyo.Constraint
            The pyomo constraint representing the upper bound relationship
        lowerBound: pyo.Constraint
            The pyomo constraint representing the lower bound relationship
        X: pyo.Var | pyo.Expression
            The Pyomo variable or expression representing "X" in this relationship.
    """
    if relationshipBaseName is None:
        Aname = str(A)
        relationshipBaseName = f"{Aname}_ConditionalLeq"

    upperBoundName = f"{relationshipBaseName}_upperBound"
    lowerBoundName = f"{relationshipBaseName}_lowerBound"

    if isinstance(alpha, (int, float)) and itrSet is not None:
        alpha = {idx: alpha for idx in itrSet}
    if isinstance(epsilon, (int, float)) and itrSet is not None:
        epsilon = {idx: epsilon for idx in itrSet}

    if X is None:
        Xname = f"{relationshipBaseName}_X"
        if itrSet is None:
            setattr(model, Xname, pyo.Var(domain=pyo.Binary))
            X = getattr(model, Xname)
        else:
            setattr(model, Xname, pyo.Var(itrSet, domain=pyo.Binary))
            X = getattr(model, Xname)

    def GetBounds(A, A_bounds, idx=None):
        """Get the bounds of A based on the provided bounds."""
        if A_bounds is not None:
            if isinstance(A_bounds, dict) and idx is not None:
                Amin, Amax = A_bounds[idx]
            elif isinstance(A_bounds, tuple):
                Amin, Amax = A_bounds
            else:
                Amin, Amax = A_bounds, A_bounds
        else:
            Amin = None
            Amax = None

        if Amin is None:
            if hasattr(A, "lb"):
                Amin = A.lb
            elif hasattr(A, "__getitem__") and idx is not None:
                Amin = A[idx].lb
        if Amax is None:
            if hasattr(A, "ub"):
                Amax = A.ub
            elif hasattr(A, "__getitem__") and idx is not None:
                Amax = A[idx].ub

        assert (
            Amin is not None
        ), "Unable to determine the lower bound of A. Please provide A_bounds or ensure A has intrinsic bounds."
        assert (
            Amax is not None
        ), "Unable to determine the upper bound of A. Please provide A_bounds or ensure A has intrinsic bounds."

        return Amin, Amax

    def TransformA(A_val, alpha_val):
        """Transform A using A' = 2*alpha - A"""
        return 2 * alpha_val - A_val

    if itrSet is None:
        Amin, Amax = GetBounds(A, A_bounds)

        # Transform A using A' = 2*alpha - A
        AminPrime = TransformA(
            Amax, alpha
        )  # Note: max becomes min after transformation
        AmaxPrime = TransformA(
            Amin, alpha
        )  # Note: min becomes max after transformation

        assert AminPrime <= AmaxPrime - epsilon, "The transformed bounds are invalid."

        # X * (alpha - AminPrime) <= Aprime - AminPrime
        # Which translates to: X * (alpha - (2*alpha - Amax)) <= (2*alpha - A) - (2*alpha - Amax)
        # Simplifying: X * (Amax - alpha) <= Amax - A
        setattr(
            model, upperBoundName, pyo.Constraint(expr=X * (Amax - alpha) <= Amax - A)
        )
        upperBound = getattr(model, upperBoundName)

        # X * (AmaxPrime - (alpha - epsilon)) >= Aprime - (alpha - epsilon)
        # Which translates to: X * ((2*alpha - Amin) - (alpha - epsilon)) >= (2*alpha - A) - (alpha - epsilon)
        # Simplifying: X * (alpha - Amin + epsilon) >= alpha - A + epsilon
        setattr(
            model,
            lowerBoundName,
            pyo.Constraint(expr=X * (alpha - Amin + epsilon) >= alpha - A + epsilon),
        )
        lowerBound = getattr(model, lowerBoundName)
    else:

        def upperBoundFunc(model, *idx):
            if len(idx) == 1:
                idx = idx[0]
            Amin, Amax = GetBounds(A, A_bounds, idx)
            alpha_val = alpha[idx] if isinstance(alpha, dict) else alpha
            return X[idx] * (Amax - alpha_val) <= Amax - A[idx]

        setattr(model, upperBoundName, pyo.Constraint(itrSet, rule=upperBoundFunc))
        upperBound = getattr(model, upperBoundName)

        def lowerBoundFunc(model, *idx):
            if len(idx) == 1:
                idx = idx[0]
            Amin, Amax = GetBounds(A, A_bounds, idx)
            alpha_val = alpha[idx] if isinstance(alpha, dict) else alpha
            epsilon_val = epsilon[idx] if isinstance(epsilon, dict) else epsilon
            return (
                X[idx] * (alpha_val - Amin + epsilon_val)
                >= alpha_val - A[idx] + epsilon_val
            )

        setattr(model, lowerBoundName, pyo.Constraint(itrSet, rule=lowerBoundFunc))
        lowerBound = getattr(model, lowerBoundName)

    return (upperBound, lowerBound, X)

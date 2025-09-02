import pyomo.kernel as pmo

from typing import Union, Tuple

from ._Formulation import _Formulation


class ConditionalGeq(_Formulation):
    """
    A block to model the following relationship in MILP form:

        A < alpha  if X == 0
        A >= alpha if X == 1

    Or, more precisely:

        A <= alpha - epsilon  if X == 0
        A >= alpha            if X == 1

    where
    * A is a variable (real or integer)
    * X is a binary
    * alpha is a constant parameter
    * epsilon is a small positive constant. Not that a zero-valued epsilon is allowed but will lead to but X==0 and X==1 being valid at A == alpha. A negative value is also allowed but will lead to X==0 and X==1 being valid for alpha <= A <= alpha - epsilon.

    A graph of the relationship:

    ^ X
    1        2---3
    |       /   /
    |      /   /
    |     /   /
    |    /   /
    |   /   /
    0  0---1
    |--mn--e-a---mx---> A

    Anywhere in this parallelogram is a feasible solution to the (linear relaxation of the) relationship.

    Points:
    0: (Amin, 0)
    1: (alpha-epsilon, 0)
    2: (alpha, 1)
    3: (Amax, 1)

    Constraints:
    0-1: Fulfilled by the notion that X is binary.
    2-3: Fulfilled by the notion that X is binary.
    0-2: X <= (A - Amin)/(alpha - Amin)
    1-3: X >= (A - (alpha - epsilon))/(Amax - (alpha - epsilon))
    """

    def __init__(
        self,
        A: Union[pmo.variable, pmo.expression],
        alpha: float,
        X: Union[pmo.variable, pmo.expression] = None,
        epsilon: float = 1e-5,
        A_bounds: Tuple[float, float] = None,
    ):
        """
        Parameters
        ----------
        A: pmo.variable | pmo.expression
            The Pyomo variable or expression representing "A" in this relationship.
        alpha: float
            A float indicating the value of "alpha" in this relationship. This is the threshold that "A" must meet or exceed when "X" is 1.
        X: pmo.variable | pmo.expression (optional, Default = None)
            The Pyomo variable or expression representing "X" in this relationship. Note that if "X" is an expression, it must evaluate to a binary value in the true feasible space. If None is provided, a unique Binary variable will be generated.
        epsilon: float (optional, Default=1e-5)
            A small positive constant to ensure strict inequality when "X" is 0.
        A_bounds: Tuple[float,float] (optional, Default=None)
            A tuple indicating the minimum and maximum possible values of "A". This is required if "A" does not have intrinsic bounds already defined or if "A" is an expression.
        """
        super().__init__(["A", "X"], {"A": (A, A_bounds), "X": (X, (0, 1))})

        if X is None:
            self.X = X = pmo.variable(domain=pmo.Binary)
            self.originalVariables[1] = self.X

        Amin, Amax = self.GetBounds(A, A_bounds)

        assert (
            Amin <= Amax - epsilon
        ), "The minimum bound of A must be less than or equal to the maximum bound of A minus epsilon."

        self.registerConstraint(
            lambda A, X: X * (alpha - Amin) <= A - Amin, name="UpperBound"
        )
        self.registerConstraint(
            lambda A, X: X * (Amax - (alpha - epsilon)) >= A - (alpha - epsilon),
            name="LowerBound",
        )

    def GetBounds(self, A, A_bounds) -> Tuple[float, float]:
        """
        Get the bounds of "A" based on the provided bounds and the relationship defined by this block.

        Parameters
        ----------
        A: pmo.variable | pmo.expression
            The Pyomo variable or expression representing "A" in this relationship.
        A_bounds: Tuple[float,float]
            The minimum and maximum possible values of "A".

        Returns
        -------
        Tuple[float,float]
            The adjusted bounds for "A" based on the relationship defined by this block.
        """
        if A_bounds is not None:
            Amin, Amax = A_bounds
        else:
            Amin = None
            Amax = None

        if Amin is None:
            Amin = A.lb
        if Amax is None:
            Amax = A.ub

        assert (
            Amin is not None
        ), "Unable to determine the lower bound of A. Please provide A_bounds or ensure A has intrinsic bounds."
        assert (
            Amax is not None
        ), "Unable to determine the upper bound of A. Please provide A_bounds or ensure A has intrinsic bounds."

        return Amin, Amax

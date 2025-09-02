import pyomo.kernel as pmo
from typing import Union

from ._Formulation import _Formulation


class DoubleSidedBigM(_Formulation):
    def __init__(
        self,
        A: Union[pmo.variable, pmo.expression],
        B: Union[pmo.variable, pmo.expression],
        Bmin: float,
        Bmax: float,
        C: Union[pmo.variable, pmo.expression, float] = 0.0,
        X: Union[pmo.variable, pmo.expression] = None,
        includeUpperBounds: bool = True,
        includeLowerBounds: bool = True,
    ):
        """
        A block to model the following relationship in MILP form:

            A = X * B + C

        where
        * A is a Real number
        * B is a Real number
        * C is a Real number, binary, or parameter
        * X is a binary.

        Parameters
        ----------
        A: pmo.variable | pmo.expression
            The Pyomo variable or expression representing "A" in this relationship
        B: pmo.variable | pmo.expression
            The Pyomo variable or expression representing "B" in this relationship
        Bmin: float
            A float indicating the minimum possible value of "B"
        Bmax: float | dict
            A float indicating the maximum possible value of "B"
        C: pmo.variable | pmo.expression | float (optional, Default=0.0)
            The value of "C" in this relationship.
        X: pmo.variable | pmo.expression (optional, Default = None)
            The Pyomo variable or expression representing "X" in this relationship. Note that if "X" is an expression, it must evaluate to a binary value in the true feasible space. If None is provided, a unique Binary variable will be generated
        includeUpperBounds: bool (optional, Default=True)
            An indication of whether or not you'd like to instantiate the upper bounds of this relationship. Only mark this as False if you're certain that "A" will never be maximized.
        includeLowerBounds: bool (optional, Default=True)
            An indication of whether or not you'd like to instantiate the lower bounds of this relationship. Only mark this as False if you're certain that "A" will never be minimized.
        """
        if isinstance(C, (float, int)):
            Cbounds = (C, C)
        elif isinstance(C, pmo.variable):
            Cbounds = (C.lb, C.ub)
        else:
            Cbounds = (None, None)
        super().__init__(
            ["B", "X", "A", "C"],
            {
                "A": (A, (min(Bmin, Cbounds[0]), max(Bmax, Cbounds[1]))),
                "B": (B, (Bmin, Bmax)),
                "X": (X, (0, 1)),
                "C": (C, Cbounds),
            },
        )

        if includeLowerBounds:
            self.registerConstraint(
                lambda B, X, A, C: A >= Bmin * X + C, name="LowerBound0"
            )
            self.registerConstraint(
                lambda B, X, A, C: A >= B + Bmax * (X - 1) + C, name="LowerBound1"
            )

        if includeUpperBounds:
            self.registerConstraint(
                lambda B, X, A, C: A <= Bmax * X + C, name="UpperBound0"
            )
            self.registerConstraint(
                lambda B, X, A, C: A <= B + Bmin * (X - 1) + C, name="UpperBound1"
            )

    def Setup(self):
        super().Setup()

        Xindex = self.variableNames.index("X")

        if self.originalVariables[Xindex] is None:
            self.X = pmo.variable(domain=pmo.Binary)
            self.originalVariables[Xindex] = self.X

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

        self.Bmin = Bmin
        self.Bmax = Bmax

        if includeLowerBounds:
            self.registerConstraint(self.LowerBound0, name="LowerBound0")
            self.registerConstraint(self.LowerBound1, name="LowerBound1")

        if includeUpperBounds:
            self.registerConstraint(self.UpperBound0, name="UpperBound0")
            self.registerConstraint(self.UpperBound1, name="UpperBound1")

    def LowerBound0(self, B, X, A, C):
        return A >= self.Bmin * X + C

    def LowerBound1(self, B, X, A, C):
        return A >= B + self.Bmax * (X - 1) + C

    def UpperBound0(self, B, X, A, C):
        return A <= self.Bmax * X + C

    def UpperBound1(self, B, X, A, C):
        return A <= B + self.Bmin * (X - 1) + C

    def Setup(self):
        super().Setup()

        Xindex = self.variableNames.index("X")

        if self.originalVariables[Xindex] is None:
            self.X = pmo.variable(domain=pmo.Binary)
            self.originalVariables[Xindex] = self.X

    def eval(self):
        """
        Assuming values are loaded for B, X, and C, determine the value of A.
        """
        Bval = pmo.value(self.originalVariables[0])
        Xval = pmo.value(self.originalVariables[1])
        Cval = pmo.value(self.originalVariables[3])
        Aval = Bval * Xval + Cval
        return Aval

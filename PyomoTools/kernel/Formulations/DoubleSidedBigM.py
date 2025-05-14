import pyomo.kernel as pmo
from typing import Union

class DoubleSidedBigM(pmo.block):
    def __init__(self,
        A:Union[pmo.variable, pmo.expression],
        B:Union[pmo.variable, pmo.expression],
        Bmin:float,
        Bmax:float,
        C:Union[pmo.variable, pmo.expression, float]=0.0,
        X:Union[pmo.variable, pmo.expression]=None,
        includeUpperBounds:bool=True,
        includeLowerBounds:bool=True):
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
        model: pmo.block
            The Pyomo model you'd like to instantiate this relationship within
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
        super().__init__()

        if X is None:
            self.X = X = pmo.variable(domain=pmo.Binary)

        if includeLowerBounds:
            self.lowerBound0 = pmo.constraint(expr=A >= Bmin * X + C)
            self.lowerBound1 = pmo.constraint(expr=A >= B + Bmax*(X-1) + C)
        
        if includeUpperBounds:
            self.upperBound0 = pmo.constraint(expr=A <= Bmax * X + C)
            self.upperBounds1 = pmo.constraint(expr=A <= B + Bmin*(X-1) + C)

        
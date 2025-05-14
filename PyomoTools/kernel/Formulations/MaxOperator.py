import pyomo.kernel as pmo
from typing import Union,Tuple
import numpy as np

class MaxOperator(pmo.block):
    def __init__(self,
        A:Union[pmo.variable, pmo.expression],
        B:Union[pmo.variable, pmo.expression],
        C:Union[pmo.variable, pmo.expression],
        bBounds:Tuple[float,float]=None,
        cBounds:Tuple[float,float]=None,
        Y:pmo.variable=None,
        allowMaximizationPotential:bool=True):
        """
        A function to model the following relationship in MILP or LP form:

            A = max(B,C)

        Parameters
        ----------
        A: pmo.variable | pmo.expression
            The Pyomo variable or expression representing "A" in this relationship
        B: pmo.variable | pmo.expression
            The Pyomo variable or expression representing "B" in this relationship
        C: pmo.variable | pmo.expression
            The Pyomo variable or expression representing "B" in this relationship
        bBounds: tuple  (optional, Default=None)
            The minimum and maximum possible values of "B". Additionally, if allowMaximizationPotential is False, bBounds can be left as None.
        cBounds: tuple | dict (optional, Default=None)
            The minimum and maximum possible values of "C". Additionally, if allowMaximizationPotential is False, cBounds can be left as None.
        Y: pmo.variable (optional, Default=None)
            The Pyomo binary variable potentially needed for representing in this relationship. If None is provided and one is needed, a unique Binary variable will be generated, if needed.
        allowMaximizationPotential: bool (optional, Default=True)
            An indication of whether or not to configure this relationship in such a way to allow "A" to be maximized. If "A" will strictly be minimized, this relationship can simply be modeled as a convex set of two inequality constraints. But if "A" can or will be maximized, this relationship must be modeled using a Binary.
        """
        super().__init__()

        if not allowMaximizationPotential:
            self.bound0 = pmo.constraint(expr=A >= B)
            self.bound1 = pmo.constraint(expr=A >= C)

        else:
            if Y is None:
                self.Y = Y = pmo.variable(domain=pmo.Binary)

            self.M = pmo.parameter(value=np.max([np.abs(bBounds[1] - cBounds[0]),np.abs(cBounds[1] - bBounds[0])])) #The maximum difference between B and C

            self.c0 = pmo.constraint(expr=B-C <= self.M * Y)

            self.c1 = pmo.constraint(expr=C-B <= self.M * (1-Y))

            self.c2 = pmo.constraint(expr=A >= B)

            self.c3 = pmo.constraint(expr=A >= C)

            self.c4 = pmo.constraint(expr=A <= B + self.M * (1-Y))

            self.c5 = pmo.constraint(expr=A <= C + self.M * Y)

        
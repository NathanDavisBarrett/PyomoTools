import pyomo.kernel as pmo
from typing import Union, Tuple
import numpy as np

from ._Formulation import _Formulation


class MaxOperator(_Formulation):
    def __init__(
        self,
        A: Union[pmo.variable, pmo.expression],
        B: Union[pmo.variable, pmo.expression],
        C: Union[pmo.variable, pmo.expression],
        bBounds: Tuple[float, float] = None,
        cBounds: Tuple[float, float] = None,
        Y: pmo.variable = None,
        allowMaximizationPotential: bool = True,
    ):
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
        vars = ["B", "C", "A"]
        varInfo = {
            "A": (A, (max(bBounds[0], cBounds[0]), max(bBounds[1], cBounds[1]))),
            "B": (B, bBounds),
            "C": (C, cBounds),
        }
        if allowMaximizationPotential:
            vars.append("Y")
            varInfo["Y"] = (Y, (0, 1))

        super().__init__(vars, varInfo)

        if not allowMaximizationPotential:
            self.registerConstraint(lambda B, C, A: A >= B)
            self.registerConstraint(lambda B, C, A: A >= C)

        else:
            if Y is None:
                self.Y = Y = pmo.variable(domain=pmo.Binary)

                self.originalVariables[3] = self.Y

            bigM = np.max(
                [np.abs(bBounds[1] - cBounds[0]), np.abs(cBounds[1] - bBounds[0])]
            )  # The maximum difference between B and C

            self.registerConstraint(
                lambda B, C, A, Y, M=bigM: B - C <= M * Y,
            )
            self.registerConstraint(
                lambda B, C, A, Y, M=bigM: C - B <= M * (1 - Y),
            )
            self.registerConstraint(
                lambda B, C, A, Y: A >= B,
            )
            self.registerConstraint(
                lambda B, C, A, Y: A >= C,
            )
            self.registerConstraint(
                lambda B, C, A, Y, M=bigM: A <= B + M * (1 - Y),
            )
            self.registerConstraint(
                lambda B, C, A, Y, M=bigM: A <= C + M * Y,
            )

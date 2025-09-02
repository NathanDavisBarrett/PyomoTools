import pyomo.kernel as pmo
from typing import Union

from ._Formulation import _Formulation


class OrOperator(_Formulation):
    def __init__(
        self,
        A: Union[pmo.variable, pmo.expression],
        B: Union[pmo.variable, pmo.expression],
        C: Union[pmo.variable, pmo.expression],
    ):
        """
        A function to model the following relationship in MILP or LP form:

            A = B || C

        This is accomplished by the following constraints:

            A <= B + C
            A >= B
            A >= C

        Parameters
        ----------
        A: pmo.variable | pmo.expression
            The Pyomo variable or expression representing "A" in this relationship. Note that "A" should either be or evaluate to a binary value (0 or 1).
        B: pmo.variable | pmo.expression
            The Pyomo variable or expression representing "B" in this relationship. Note that "B" should either be or evaluate to a binary value (0 or 1).
        C: pmo.variable | pmo.expression
            The Pyomo variable or expression representing "C" in this relationship. Note that "C" should either be or evaluate to a binary value (0 or 1).
        """
        super().__init__(
            ["B", "C", "A"], {"A": (A, (0, 1)), "B": (B, (0, 1)), "C": (C, (0, 1))}
        )

        self.registerConstraint(
            lambda B, C, A: A <= B + C,
        )
        self.registerConstraint(
            lambda B, C, A: A >= B,
        )
        self.registerConstraint(
            lambda B, C, A: A >= C,
        )

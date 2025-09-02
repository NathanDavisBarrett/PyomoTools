import pyomo.kernel as pmo
from typing import Union, Tuple

from .MaxOperator import MaxOperator


class MinOperator(MaxOperator):
    def __init__(
        self,
        A: Union[pmo.variable, pmo.expression],
        B: Union[pmo.variable, pmo.expression],
        C: Union[pmo.variable, pmo.expression],
        bBounds: Tuple[float, float] = None,
        cBounds: Tuple[float, float] = None,
        Y: pmo.variable = None,
        allowMinimizationPotential: bool = True,
    ):
        """
        A function to model the following relationship in MILP or LP form:

            A = min(B,C)

        Note that this is equivalent to

            -A = max(-B,-C)

        which is exactly how this will be implemented in the model.


        Parameters
        ----------
        A: pmo.variable | pmo.expression
            The Pyomo variable or expression representing "A" in this relationship
        B: pmo.variable | pmo.expression
            The Pyomo variable or expression representing "B" in this relationship
        C: pmo.variable | pmo.expression
            The Pyomo variable or expression representing "B" in this relationship
        bBounds: tuple  (optional, Default=None)
            The minimum and maximum possible values of "B". Additionally, if allowMinimizationPotential is False, bBounds can be left as None.
        cBounds: tuple | dict (optional, Default=None)
            The minimum and maximum possible values of "C". Additionally, if allowMinimizationPotential is False, cBounds can be left as None.
        Y: pmo.variable (optional, Default=None)
            The Pyomo binary variable potentially needed for representing in this relationship. If None is provided and one is needed, a unique Binary variable will be generated, if needed.
        allowMaximizationPotential: bool (optional, Default=True)
            An indication of whether or not to configure this relationship in such a way to allow "A" to be minimized. If "A" will strictly be maximized, this relationship can simply be modeled as a convex set of two inequality constraints. But if "A" can or will be minimized, this relationship must be modeled using a Binary.
        """
        opposite_bBounds = None
        opposite_cBounds = None

        if bBounds is not None:
            opposite_bBounds = (
                -bBounds[1] if bBounds[1] is not None else None,
                -bBounds[0] if bBounds[0] is not None else None,
            )
        if cBounds is not None:
            opposite_cBounds = (
                -cBounds[1] if cBounds[1] is not None else None,
                -cBounds[0] if cBounds[0] is not None else None,
            )

        super().__init__(
            A=-A,
            B=-B,
            C=-C,
            bBounds=opposite_bBounds,
            cBounds=opposite_cBounds,
            Y=Y,
            allowMaximizationPotential=allowMinimizationPotential,
        )

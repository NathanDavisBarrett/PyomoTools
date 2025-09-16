import pyomo.kernel as pmo

from typing import Iterable, Union


class Conic(pmo.block):
    """
    A block for enforcing conic constraints.

    A conic relationship is of the form:

        ||x||_o <= r

    where o is the order of the conic constraint (a positive integer), ||.||_o is the L-o norm, x is an iterable of either expressions or variables, and r >= 0 is an expression or variable.

    In practice, this constraint is implemented as:

        sum(x_i^o for x_i in x) <= r^o        if o is even

        sum(x_i^o for x_i in x) <= r^o   AND    if o is odd
        sum(-x_i^o for x_i in x) <= r^o         ...
    """

    def __init__(
        self,
        x: Iterable[Union[pmo.variable, pmo.expression, float]],
        r: Union[pmo.variable, pmo.expression, float],
        order,
        explicit_r_nonnegativity: bool = True,
    ):
        """
        Parameters
        ----------
        x : Iterable[Union[pmo.variable, pmo.expression, float]]
            The left-hand side of the conic constraint.
        r : Union[pmo.variable, pmo.expression, float]
            The right-hand side of the conic constraint.
        order : int, optional
            The order of the conic constraint
        explicit_r_nonnegativity : bool, optional
            Whether to explicitly enforce the non-negativity of r when the order is odd. Default is True.
        """
        super().__init__()

        assert isinstance(order, int) and order > 0, "Order must be a positive integer."

        if explicit_r_nonnegativity:
            self.r_nonnegativity = pmo.constraint(expr=r >= 0)

        self.conic_constraint = pmo.constraint(
            expr=sum(xi**order for xi in x) <= r**order
        )
        if order % 2 != 0:
            self.conic_constraint_neg = pmo.constraint(
                expr=sum(-(xi**order) for xi in x) <= r**order
            )

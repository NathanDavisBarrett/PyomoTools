import pyomo.kernel as pmo
from typing import Union

class AndOperator(pmo.block):
    def __init__(self,
        A:Union[pmo.variable, pmo.expression],
        B:Union[pmo.variable, pmo.expression],
        C:Union[pmo.variable, pmo.expression]):
        """
        A function to model the following relationship in MILP or LP form:

            A = B && C

        This is accomplished by the following constraints:

            A >= B + C - 1
            A <= B
            A <= C

        Parameters
        ----------
        A: pmo.variable | pmo.expression
            The Pyomo variable or expression representing "A" in this relationship. Note that "A" should either be or evaluate to a binary value (0 or 1).
        B: pmo.variable | pmo.expression
            The Pyomo variable or expression representing "B" in this relationship. Note that "B" should either be or evaluate to a binary value (0 or 1).
        C: pmo.variable | pmo.expression
            The Pyomo variable or expression representing "C" in this relationship. Note that "C" should either be or evaluate to a binary value (0 or 1).
        """
        super().__init__()

        self.c0 = pmo.constraint(expr=A >= B + C - 1)
        self.c1 = pmo.constraint(expr=A <= B)
        self.c2 = pmo.constraint(expr=A <= C)



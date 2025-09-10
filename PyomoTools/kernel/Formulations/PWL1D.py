from ...base.Formulations.PWL1D import PWL1DParameters, PWL1DType

from typing import Union

import pyomo.kernel as pmo


class PWL1D(pmo.block):
    """
    A block to model a piecewise linear function in MILP or LP form.

    The function creates the appropriate constraints based on the PWL type:
    - LINEAR: Single linear equality constraint
    - CONVEX: Linear inequalities for convex hull
    - CONCAVE: Linear inequalities for concave envelope
    - GENERAL: SOS2 formulation with weights

    Parameters
    ----------
    params: PWL1DParameters | Dict[Any, PWL1DParameters]
        The parameters defining the piecewise linear function.
    xVar: pyo.variable | pyo.expression
        The Pyomo variable or expression representing the x-coordinate
    yVar: pyo.variable | pyo.expression
        The Pyomo variable or expression representing the y-coordinate
    """

    def __init__(
        self,
        params: PWL1DParameters,
        xVar: Union[pmo.variable, pmo.expression, float],
        yVar: Union[pmo.variable, pmo.expression, float],
    ):
        super().__init__()
        self.params = params

        pwl_type = self.params.pwl_type

        if pwl_type == PWL1DType.LINEAR:
            self._init_linear(xVar, yVar)
        elif pwl_type == PWL1DType.CONVEX:
            self._init_convex(xVar, yVar)
        elif pwl_type == PWL1DType.CONCAVE:
            self._init_concave(xVar, yVar)
        elif pwl_type == PWL1DType.GENERAL:
            self._init_general(xVar, yVar)
        else:
            raise ValueError(f"Unsupported PWL type: {pwl_type}")

        self._enforce_x_bounds(xVar)

    def _enforce_x_bounds(self, xVar):
        if self.params.includeLB_x:
            self.x_lb = pmo.constraint(expr=xVar >= self.params.points[0][0])

        if self.params.includeUB_x:
            self.x_ub = pmo.constraint(expr=xVar <= self.params.points[-1][0])

    def _init_linear(self, xVar, yVar):
        p1, p2 = self.params.points[0], self.params.points[1]

        self.linearEquality = pmo.constraint(
            (yVar - p1[1]) * (p2[0] - p1[0]) == (xVar - p1[0]) * (p2[1] - p1[1])
        )

    def _init_convex(self, xVar, yVar):
        self.convexInequalities = pmo.constraint_dict()
        for i in range(1, len(self.params.points)):
            p1, p2 = self.params.points[i - 1], self.params.points[i]
            dx = p2[0] - p1[0]
            if dx >= 0:
                self.convexInequalities[i] = pmo.constraint(
                    (yVar - p1[1]) * dx >= (xVar - p1[0]) * (p2[1] - p1[1])
                )
            else:
                self.convexInequalities[i] = pmo.constraint(
                    (yVar - p1[1]) * dx <= (xVar - p1[0]) * (p2[1] - p1[1])
                )

    def _init_concave(self, xVar, yVar):
        self.concaveInequalities = pmo.constraint_dict()
        for i in range(1, len(self.params.points)):
            p1, p2 = self.params.points[i - 1], self.params.points[i]
            dx = p2[0] - p1[0]
            if dx >= 0:
                self.concaveInequalities[i] = pmo.constraint(
                    (yVar - p1[1]) * dx <= (xVar - p1[0]) * (p2[1] - p1[1])
                )
            else:
                self.concaveInequalities[i] = pmo.constraint(
                    (yVar - p1[1]) * dx >= (xVar - p1[0]) * (p2[1] - p1[1])
                )

    def _init_general(self, xVar, yVar):
        self.weights = pmo.variable_list(
            [
                pmo.variable(domain=pmo.NonNegativeReals)
                for _ in range(self.params.num_points)
            ]
        )

        self.weightSumConstraint = pmo.constraint(
            sum(self.weights[i] for i in range(self.params.num_points)) == 1
        )

        self.sos2Constraint = pmo.sos2(self.weights)

        self.xValueConstraint = pmo.constraint(
            xVar
            == sum(
                self.weights[i] * self.params.points[i][0]
                for i in range(self.params.num_points)
            )
        )

        self.yValueConstraint = pmo.constraint(
            yVar
            == sum(
                self.weights[i] * self.params.points[i][1]
                for i in range(self.params.num_points)
            )
        )

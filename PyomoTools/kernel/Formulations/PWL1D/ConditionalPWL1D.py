from ....base.Formulations.PWL1D import PWL1DParameters, PWL1DType
from ..DoubleSidedBigM import DoubleSidedBigM as DSBM

import pyomo.kernel as pmo
from enum import Enum
from typing import Tuple, Union


class RelationshipType(Enum):
    """
    Enum to define the type of relationship for the conditional piecewise linear function.
    """

    LEQ = 1  # Less than or equal to
    GEQ = 2  # Greater than or equal to
    EQ = 3  # Equal to


class XBoundOption(Enum):
    """
    Enum to define the interpretation of the bounds for the x variable in the conditional piecewise linear function.
    """

    POINT_BOUND = 1  # Use the bounds of the points defining the PWL function
    VARIABLE_BOUND = 2  # Use the bounds of the x variable itself
    TIGHTEST_BOUND = 3  # Use the tightest bound between the points and the variable


class ConditionalPWL1D(pmo.block):
    """
    A block to model a conditional piecewise linear function in MILP or LP form.

    yVAr = PWL(xVar) if condition == 1, else 0

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
    condition: pyo.variable | pyo.expression
        A binary variable that indicates whether the PWL function is active (1) or inactive (0)
    xBoundOption: XBoundOption
        The interpretation of the bounds for the x variable in the conditional piecewise linear function.
    """

    def __init__(
        self,
        params: PWL1DParameters,
        xVar: Union[pmo.variable, pmo.expression, float],
        yVar: Union[pmo.variable, pmo.expression, float],
        condition: Union[pmo.variable, pmo.expression],
        xBoundOption: XBoundOption = XBoundOption.TIGHTEST_BOUND,
    ):
        if not params.includeLB_x or not params.includeUB_x:
            raise ValueError(
                "For conditional PWL functions, xVar must have both lower and upper bounds defined."
            )
        super().__init__()
        self.params = params
        self.x_bounds = self._determine_x_bounds(xVar, xBoundOption)

        pwl_type = self.params.pwl_type

        if pwl_type == PWL1DType.LINEAR:
            self._init_linear(xVar, yVar, condition)
        elif pwl_type == PWL1DType.CONVEX:
            self._init_convex(xVar, yVar, condition)
        elif pwl_type == PWL1DType.CONCAVE:
            self._init_concave(xVar, yVar, condition)
        elif pwl_type == PWL1DType.GENERAL:
            self._init_general(xVar, yVar, condition)
        else:
            raise ValueError(f"Unsupported PWL type: {pwl_type}")

    def _determine_x_bounds(
        self, xVar, xBoundOption: XBoundOption
    ) -> Tuple[float, float]:
        if xBoundOption == XBoundOption.POINT_BOUND:
            return self.params._point_box[0]
        elif xBoundOption == XBoundOption.VARIABLE_BOUND:
            if xVar.lb is None or xVar.ub is None:
                raise ValueError(
                    "For VARIABLE_BOUND, xVar must have both lower and upper bounds defined."
                )
            return xVar.lb, xVar.ub
        elif xBoundOption == XBoundOption.TIGHTEST_BOUND:
            x_lb = (
                max(xVar.lb, self.params._point_box[0][0])
                if xVar.lb is not None
                else self.params._point_box[0][0]
            )
            x_ub = (
                min(xVar.ub, self.params._point_box[0][1])
                if xVar.ub is not None
                else self.params._point_box[0][1]
            )
            if x_lb > x_ub:
                raise ValueError(
                    "The tightest bounds for xVar are inconsistent. Check the bounds of xVar and the PWL points."
                )
            return x_lb, x_ub
        else:
            raise ValueError(f"Unsupported XBoundOption: {xBoundOption}")

    def _generate_non_vertical_conditional_relation(
        self, xVar, yVar, condition, slope, intercept, inequality_type: RelationshipType
    ) -> pmo.block:
        if slope >= 0:
            min_value = slope * self.x_bounds[0] + intercept
            max_value = slope * self.x_bounds[1] + intercept
        else:
            min_value = slope * self.x_bounds[1] + intercept
            max_value = slope * self.x_bounds[0] + intercept

        if inequality_type == RelationshipType.LEQ:
            includeUpperBounds = True
            includeLowerBounds = False
        elif inequality_type == RelationshipType.GEQ:
            includeUpperBounds = False
            includeLowerBounds = True
        elif inequality_type == RelationshipType.EQ:
            includeUpperBounds = True
            includeLowerBounds = True
        else:
            raise ValueError(f"Unsupported RelationshipType: {inequality_type}")

        return DSBM(
            A=yVar,
            B=slope * xVar + intercept,
            Bmin=min_value,
            Bmax=max_value,
            X=condition,
            includeUpperBounds=includeUpperBounds,
            includeLowerBounds=includeLowerBounds,
        )

    def _init_linear(self, xVar, yVar, condition):
        if self.params._point_box[0][1] - self.params._point_box[0][0] < 1e-8:
            if self.x_bounds[1] - self.x_bounds[0] < 1e-8:
                self.vertical_linear_x_value_constraint = pmo.constraint(
                    xVar == self.x_bounds[0]
                )
            else:
                self.vertical_linear_x_value_ub = pmo.constraint(
                    xVar
                    <= self.x_bounds[1]
                    + (self.params._point_box[0][1] - self.x_bounds[1]) * condition
                )
                self.vertical_linear_x_value_lb = pmo.constraint(
                    xVar
                    >= self.x_bounds[0]
                    + (self.params._point_box[0][0] - self.x_bounds[0]) * condition
                )
            self.vertical_linear_y_value_ub = pmo.constraint(
                yVar <= self.params._point_box[1][1] * condition
            )
            self.vertical_linear_y_value_lb = pmo.constraint(
                yVar >= self.params._point_box[1][0] * condition
            )
        else:
            slope = (self.params.points[1][1] - self.params.points[0][1]) / (
                self.params.points[1][0] - self.params.points[0][0]
            )
            intercept = self.params.points[0][1] - slope * self.params.points[0][0]
            self.conditional_linear = self._generate_non_vertical_conditional_relation(
                xVar, yVar, condition, slope, intercept, RelationshipType.EQ
            )

    def _generate_planar_constraint(
        self, xVar, yVar, condition, points, inequality_type: RelationshipType
    ):
        if len(points) != 3:
            raise ValueError(
                "Exactly three points are required to define a planar constraint."
            )
        for p in points:
            if len(p) != 3:
                raise ValueError(
                    "Each point must be a tuple of (xVar, condition, yVar)."
                )

        (px1, py1, pz1), (px2, py2, pz2), (px3, py3, pz3) = points

        # Calculate the plane coefficients
        A = (py2 - py1) * (pz3 - pz1) - (py3 - py1) * (pz2 - pz1)
        B = (px3 - px1) * (pz2 - pz1) - (px2 - px1) * (pz3 - pz1)
        C = (px2 - px1) * (py3 - py1) - (px3 - px1) * (py2 - py1)
        D = -(A * px1 + B * py1 + C * pz1)

        # Final answer will always be of the form A*xVar + B*condition + C*yVar + D <= 0 (regardless of the inequality type).

        if inequality_type == RelationshipType.LEQ:
            test_point = (px1, py1, pz1 + 1)  # Should be outside the feasible region
            if A * test_point[0] + B * test_point[1] + C * test_point[2] + D < 0:
                A, B, C, D = -A, -B, -C, -D  # Flip the inequality
        elif inequality_type == RelationshipType.GEQ:
            test_point = (px1, py1, pz1 - 1)  # Should be outside the feasible region
            if A * test_point[0] + B * test_point[1] + C * test_point[2] + D < 0:
                A, B, C, D = -A, -B, -C, -D  # Flip the inequality
        elif inequality_type == RelationshipType.EQ:
            return pmo.constraint(A * xVar + B * condition + C * yVar + D == 0)

        return pmo.constraint(A * xVar + B * condition + C * yVar + D <= 0)

    def _init_convex(self, xVar, yVar, condition):
        self.convexInequalities = pmo.constraint_dict()
        horizontal_segment_found = False
        for i in range(1, len(self.params.points)):
            (px1, py1), (px2, py2) = self.params.points[i - 1], self.params.points[i]

            if abs(px2 - px1) < 1e-8:
                # Vertical segment
                if i == 1:
                    lb = self.x_bounds[0]
                    # This is the first segment so it will be px1 <= xVar
                    self.convexInequalities[i] = pmo.constraint(
                        lb + (px1 - lb) * condition <= xVar
                    )
                elif i == len(self.params.points) - 1:
                    ub = self.x_bounds[1]
                    # This is the last segment so it will be xVar <= px2
                    self.convexInequalities[i] = pmo.constraint(
                        xVar <= ub + (px2 - ub) * condition
                    )
                else:
                    raise ValueError(
                        "Conditional convex functions cannot have vertical segments in the middle of the function."
                    )
            else:
                slope = (py2 - py1) / (px2 - px1)

                if abs(slope) < 1e-8:
                    horizontal_segment_found = True

                if slope <= 0:
                    # Form a plane with the minimum x-value corner point
                    self.convexInequalities[i] = self._generate_planar_constraint(
                        xVar,
                        yVar,
                        condition,
                        [
                            (px1, 1, py1),
                            (px2, 1, py2),
                            (self.x_bounds[0], 0, 0),
                        ],
                        RelationshipType.GEQ,
                    )
                else:
                    # Form a plane with the maximum x-value corner point
                    self.convexInequalities[i] = self._generate_planar_constraint(
                        xVar,
                        yVar,
                        condition,
                        [
                            (px1, 1, py1),
                            (px2, 1, py2),
                            (self.x_bounds[1], 0, 0),
                        ],
                        RelationshipType.GEQ,
                    )
        if not horizontal_segment_found:
            # Add the flat, bottom-most plane. (i.e. a "horizontal segment" from the lowest y-valued point)
            minimum_y_value = min(p[1] for p in self.params.points)
            self.convexInequalities["bottom"] = self._generate_planar_constraint(
                xVar,
                yVar,
                condition,
                [
                    (self.x_bounds[0], 0, 0),
                    (self.x_bounds[1], 0, 0),
                    (self.x_bounds[0], 1, minimum_y_value),
                ],
                RelationshipType.GEQ,
            )

    def _init_concave(self, xVar, yVar, condition):
        self.concaveInequalities = pmo.constraint_dict()
        horizontal_segment_found = False
        for i in range(1, len(self.params.points)):
            (px1, py1), (px2, py2) = self.params.points[i - 1], self.params.points[i]

            if abs(px2 - px1) < 1e-8:
                # Vertical segment
                if i == 1:
                    lb = self.x_bounds[0]
                    # This is the first segment so it will be px1 <= xVar
                    self.concaveInequalities[i] = pmo.constraint(
                        lb + (px1 - lb) * condition <= xVar
                    )
                elif i == len(self.params.points) - 1:
                    ub = self.x_bounds[1]
                    # This is the last segment so it will be xVar <= px2
                    self.concaveInequalities[i] = pmo.constraint(
                        xVar <= ub + (px2 - ub) * condition
                    )
                else:
                    raise ValueError(
                        "Conditional concave functions cannot have vertical segments in the middle of the function."
                    )
            else:
                slope = (py2 - py1) / (px2 - px1)

                if abs(slope) < 1e-8:
                    horizontal_segment_found = True

                if slope >= 0:
                    # Form a plane with the minimum x-value corner point
                    self.concaveInequalities[i] = self._generate_planar_constraint(
                        xVar,
                        yVar,
                        condition,
                        [
                            (px1, 1, py1),
                            (px2, 1, py2),
                            (self.x_bounds[0], 0, 0),
                        ],
                        RelationshipType.LEQ,
                    )
                else:
                    # Form a plane with the maximum x-value corner point
                    self.concaveInequalities[i] = self._generate_planar_constraint(
                        xVar,
                        yVar,
                        condition,
                        [
                            (px1, 1, py1),
                            (px2, 1, py2),
                            (self.x_bounds[1], 0, 0),
                        ],
                        RelationshipType.LEQ,
                    )
        if not horizontal_segment_found:
            # Add the flat, top-most plane. (i.e. a "horizontal segment" from the highest y-valued point)
            maximum_y_value = max(p[1] for p in self.params.points)
            self.concaveInequalities["top"] = self._generate_planar_constraint(
                xVar,
                yVar,
                condition,
                [
                    (self.x_bounds[0], 0, 0),
                    (self.x_bounds[1], 0, 0),
                    (self.x_bounds[0], 1, maximum_y_value),
                ],
                RelationshipType.LEQ,
            )

    def _init_general(self, xVar, yVar, condition):
        self.weights = pmo.variable_list(
            [
                pmo.variable(domain=pmo.NonNegativeReals)
                for _ in range(self.params.num_points)
            ]
        )

        self.weightSumConstraint = pmo.constraint(
            sum(self.weights[i] for i in range(self.params.num_points)) == condition
        )

        self.sos2Constraint = pmo.sos2(self.weights)

        self.xValueUB = pmo.constraint(
            xVar
            <= sum(
                self.weights[i] * self.params.points[i][0]
                for i in range(self.params.num_points)
            )
            + (1 - condition) * self.x_bounds[1]
        )
        self.xValueLB = pmo.constraint(
            xVar
            >= sum(
                self.weights[i] * self.params.points[i][0]
                for i in range(self.params.num_points)
            )
            + (1 - condition) * self.x_bounds[0]
        )

        self.yValueConstraint = pmo.constraint(
            yVar
            == sum(
                self.weights[i] * self.params.points[i][1]
                for i in range(self.params.num_points)
            )
        )

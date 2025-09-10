"""
One-Dimensional Piecewise Linear Function Module

This module provides classes and functionality for defining, analyzing, and working
with one-dimensional piecewise linear (PWL) functions. It includes tools for:

- Defining PWL functions from points, slopes, or simple linear/constant functions
- Automatically determining PWL function types (convex, concave, general, linear)
- Validating PWL function constraints and bounds
- Visualizing PWL functions

The module is designed to support optimization problems that involve piecewise
linear relationships, particularly in the context of Pyomo modeling.

Classes:
    PWL1DType: Enumeration of possible PWL function types
    PWL1DParameters: Main class for defining and analyzing PWL functions

Author: PyomoTools
"""

from dataclasses import dataclass
from enum import Enum
from warnings import warn

from typing import List, Tuple
from functools import cached_property


class PWL1DType(Enum):
    """
    Enumeration of possible types for one-dimensional piecewise linear functions.

    This classification is important for optimization as different PWL types require
    different mathematical formulations and constraint structures.

    Attributes:
        CONVEX (1): PWL function with non-decreasing slopes between segments.
                   Requires lower bound for proper formulation.
        CONCAVE (2): PWL function with non-increasing slopes between segments.
                    Requires upper bound for proper formulation.
        GENERAL (3): PWL function that is neither purely convex nor concave.
                    Requires both upper and lower bounds.
        LINEAR (4): PWL function with constant slope (straight line).
                   Can be formulated as a simple linear constraint.
        UNABLE_TO_DETERMINE (5): PWL function type cannot be determined due to
                               invalid bound specifications.
    """

    CONVEX = 1
    CONCAVE = 2
    GENERAL = 3
    LINEAR = 4
    UNABLE_TO_DETERMINE = 5


@dataclass
class PWL1DParameters:
    """
    Parameters for defining a one-dimensional piecewise linear function.

    This class provides a comprehensive interface for creating and analyzing PWL
    functions from various input formats. It automatically validates inputs,
    determines function properties, and provides utility methods for analysis.

    Attributes:
        points (List[Tuple[float, float]]): List of (x, y) coordinate pairs that
            define the PWL function. Points are automatically sorted by x-value.
        includeUB_y (bool): Whether to include an upper bound constraint. Defaults
            to True. Required for concave and general PWL functions.
        includeLB_y (bool): Whether to include a lower bound constraint. Defaults
            to True. Required for convex and general PWL functions.
        includeUB_x (bool): Whether to explicitly include a constraint for the upper bound of the x variable. Defaults
            to True. Required to be True for general PWL functions.
        includeLB_x (bool): Whether to explicitly include a constraint for the lower bound of the x variable. Defaults
            to True. Required to be True for general PWL functions.

    Raises:
        ValueError: If fewer than 2 points are provided, or if both includeUB_y
                   and includeLB_y are False.

    Example:
        >>> # Create a simple triangular PWL function
        >>> params = PWL1DParameters([(0, 0), (1, 2), (2, 0)])
        >>> print(params.pwl_type)  # Will determine if convex/concave/general

        >>> # Create from progressive slopes
        >>> params = PWL1DParameters.from_progressive_slopes(
        ...     xValues=[0, 1, 2], slopes=[2, -2], startY=0
        ... )
    """

    points: List[Tuple[float, float]]  # List of (x,y) points defining the PWL function
    includeUB_y: bool = True
    includeLB_y: bool = True
    includeUB_x: bool = True
    includeLB_x: bool = True

    def __post_init__(self):
        """
        Post-initialization validation and setup.

        Automatically called after dataclass initialization to:
        1. Validate that at least 2 points are provided
        2. Sort points by x-coordinate to ensure proper ordering
        3. Validate that at least one bound type is included

        Raises:
            ValueError: If fewer than 2 points provided or if both bounds are excluded.
        """
        if len(self.points) < 2:
            raise ValueError(
                "At least two points are required to define a PWL function."
            )
        # Sort points by x-coordinate to ensure proper ordering for analysis
        self.points.sort(key=lambda pt: pt[0])
        if not (self.includeLB_y or self.includeUB_y):
            raise ValueError("At least one of includeLB_y or includeUB_y must be True.")

    @classmethod
    def from_progressive_slopes(
        cls,
        xValues: List[float],
        slopes: List[float],
        startY: float = 0.0,
        includeUB_y: bool = True,
        includeLB_y: bool = True,
        includeUB_x: bool = True,
        includeLB_x: bool = True,
    ):
        """
        Create a PWL function from a series of x-values and corresponding slopes.

        This method constructs a PWL function by specifying x-coordinates where
        the slope changes and the slope values for each segment. The y-values
        are computed progressively starting from a given initial y-value.

        Args:
            xValues (List[float]): X-coordinates defining segment boundaries.
                Must have one more element than slopes (n+1 x-values for n slopes).
            slopes (List[float]): Slope values for each segment between consecutive
                x-values. Length must be len(xValues) - 1.
            startY (float, optional): Y-coordinate at the first x-value. Defaults to 0.0.
            includeUB_y (bool, optional): Include upper bound constraint. Defaults to True.
            includeLB_y (bool, optional): Include lower bound constraint. Defaults to True.
            includeUB_x (bool, optional): Whether to explicitly include a constraint for the upper bound of the x variable. Defaults to True.
            includeLB_x (bool, optional): Whether to explicitly include a constraint for the lower bound of the x variable. Defaults to True.

        Returns:
            PWL1DParameters: Configured PWL function parameters.

        Raises:
            AssertionError: If len(xValues) != len(slopes) + 1.

        Example:
            >>> # Create a PWL function with slopes [2, -1, 0.5] between x-values [0, 1, 2, 3]
            >>> params = PWL1DParameters.from_progressive_slopes(
            ...     xValues=[0, 1, 2, 3],
            ...     slopes=[2, -1, 0.5],
            ...     startY=1.0
            ... )
            >>> # Results in points: [(0, 1), (1, 3), (2, 2), (3, 2.5)]
        """
        assert (
            len(xValues) == len(slopes) + 1
        ), "Number of xValues must be one more than number of slopes: xValues defines neighboring x bounds within which the slopes apply."

        # Start with the initial point
        points = [(xValues[0], startY)]

        # Progressively calculate each subsequent point using the slopes
        for i in range(len(slopes)):
            deltaX = xValues[i + 1] - xValues[i]  # Width of current segment
            deltaY = slopes[i] * deltaX  # Change in y for this segment
            newY = points[-1][1] + deltaY  # New y-coordinate
            points.append((xValues[i + 1], newY))

        return cls(
            points=points,
            includeUB_y=includeUB_y,
            includeLB_y=includeLB_y,
            includeUB_x=includeUB_x,
            includeLB_x=includeLB_x,
        )

    @classmethod
    def from_constant(
        cls, value, includeUB_y=True, includeLB_y=True, lb_x=None, ub_x=None
    ):
        """
        Create a PWL function representing a constant value.

        Creates a horizontal line (zero slope) PWL function with the specified
        constant value. This is useful for representing fixed costs or constraints.

        Args:
            value (float): The constant y-value for the function.
            includeUB_y (bool, optional): Include upper bound constraint. Defaults to True.
            includeLB_y (bool, optional): Include lower bound constraint. Defaults to True.
            lb_x (float, optional): Lower bound for the x variable. If None, no explicit lower bound is set. Defaults to None.
            ub_x (float, optional): Upper bound for the x variable. If None, no explicit upper bound is set. Defaults to None.

        Returns:
            PWL1DParameters: PWL function with constant value across x-domain [0, 1].

        Example:
            >>> # Create a constant function at y=5
            >>> params = PWL1DParameters.from_constant(5.0)
            >>> # Results in points: [(0, 5), (1, 5)]
        """
        if ub_x is not None and lb_x is not None:
            assert ub_x > lb_x, "ub_x must be greater than lb_x."

        if lb_x is None:
            if ub_x is not None:
                lb_x = min(ub_x * 0.9, ub_x - 1)  # Ensure lb_x is less than ub_x
            else:
                lb_x = 0
            includeLB_x = False
        else:
            includeLB_x = True

        if ub_x is None:
            ub_x = max(lb_x * 1.1, lb_x + 1)  # Ensure ub_x is greater than lb_x
            includeUB_x = False
        else:
            includeUB_x = True

        return cls(
            points=[(lb_x, value), (ub_x, value)],
            includeUB_y=includeUB_y,
            includeLB_y=includeLB_y,
            includeUB_x=includeUB_x,
            includeLB_x=includeLB_x,
        )

    @classmethod
    def from_linear(
        cls,
        slope,
        intercept,
        includeUB_y=True,
        includeLB_y=True,
        lb_x=None,
        ub_x=None,
    ):
        """
        Create a PWL function representing a simple linear function.

        Creates a straight line PWL function with the specified slope and y-intercept.
        The function is defined over the domain [0, 1] by default.

        Args:
            slope (float): The slope (rate of change) of the linear function.
            intercept (float): The y-intercept (value when x=0).
            includeUB_y (bool, optional): Include upper bound constraint. Defaults to True.
            includeLB_y (bool, optional): Include lower bound constraint. Defaults to True.
            lb_x (float, optional): Lower bound for the x variable. If None, no explicit lower bound is set. Defaults to None.
            ub_x (float, optional): Upper bound for the x variable. If None, no explicit upper bound is set. Defaults to None.

        Returns:
            PWL1DParameters: PWL function representing y = slope*x + intercept.

        Example:
            >>> # Create a linear function: y = 2x + 3
            >>> params = PWL1DParameters.from_linear(slope=2, intercept=3)
            >>> # Results in points: [(0, 3), (1, 5)]
        """
        if ub_x is not None and lb_x is not None:
            assert ub_x > lb_x, "ub_x must be greater than lb_x."

        if lb_x is None:
            if ub_x is not None:
                lb_x = min(ub_x * 0.9, ub_x - 1)  # Ensure lb_x is less than ub_x
            else:
                lb_x = 0
            includeLB_x = False
        else:
            includeLB_x = True

        if ub_x is None:
            ub_x = max(lb_x * 1.1, lb_x + 1)  # Ensure ub_x is greater than lb_x
            includeUB_x = False
        else:
            includeUB_x = True

        p1 = (lb_x, slope * lb_x + intercept)  # Point at x=lb_x
        p2 = (ub_x, slope * ub_x + intercept)  # Point at x=ub_x
        return cls(
            points=[p1, p2],
            includeUB_y=includeUB_y,
            includeLB_y=includeLB_y,
            includeUB_x=includeUB_x,
            includeLB_x=includeLB_x,
        )

    @cached_property
    def num_points(self) -> int:
        """
        Get the number of points defining the PWL function.

        Returns:
            int: Number of (x, y) coordinate pairs in the function definition.
        """
        return len(self.points)

    @cached_property
    def num_segments(self) -> int:
        """
        Get the number of linear segments in the PWL function.

        Each segment connects two consecutive points, so the number of segments
        is always one less than the number of points.

        Returns:
            int: Number of linear segments (num_points - 1).
        """
        return self.num_points - 1

    @cached_property
    def pwl_type(self):
        """
        Automatically determine the type of the PWL function based on slope patterns.

        Analyzes the slopes between consecutive segments to classify the function as:
        - CONVEX: Non-decreasing slopes (suitable for lower-bound formulations)
        - CONCAVE: Non-increasing slopes (suitable for upper-bound formulations)
        - LINEAR: Constant slope (simple linear constraint)
        - GENERAL: Neither convex nor concave (requires both bounds)
        - UNABLE_TO_DETERMINE: Invalid bound configuration for the function type

        The classification uses a small tolerance (1e-7) to handle numerical precision
        issues when comparing slopes.

        Returns:
            PWL1DType: The determined function type.

        Warnings:
            Issues warnings when bound configurations are incompatible with function types:
            - Convex functions require lower bounds
            - Concave functions require upper bounds
            - General functions require both bounds
        """
        # Calculate slopes for each segment
        slopes = [
            (self.points[i + 1][1] - self.points[i][1])
            / (self.points[i + 1][0] - self.points[i][0])
            for i in range(len(self.points) - 1)
        ]

        # Check convexity: slopes should be non-decreasing (with tolerance)
        is_convex = all(
            slopes[i] <= slopes[i + 1] + 1e-7 for i in range(len(slopes) - 1)
        )

        # Check concavity: slopes should be non-increasing (with tolerance)
        is_concave = all(
            slopes[i] >= slopes[i + 1] - 1e-7 for i in range(len(slopes) - 1)
        )

        # Determine function type based on convexity/concavity and bound settings
        if not is_convex and not is_concave:
            # Neither convex nor concave - requires both bounds
            if self.includeLB_y and self.includeUB_y:
                if not (self.includeLB_x and self.includeUB_x):
                    warn(
                        "General PWL functions must include both lower and upper bounds on the x variable."
                    )
                    return PWL1DType.UNABLE_TO_DETERMINE
                return PWL1DType.GENERAL
            elif self.includeLB_y and not self.includeUB_y:
                warn(
                    "PWL functions that are neither convex nor concave must include an upper bound."
                )
                return PWL1DType.UNABLE_TO_DETERMINE
            elif not self.includeLB_y and self.includeUB_y:
                warn(
                    "PWL functions that are neither convex nor concave must include a lower bound."
                )
                return PWL1DType.UNABLE_TO_DETERMINE
        elif is_convex and not is_concave:
            # Purely convex function
            if not self.includeLB_y:
                warn("Convex PWL functions must include a lower bound.")
                return PWL1DType.UNABLE_TO_DETERMINE
            if not self.includeUB_y:
                return PWL1DType.CONVEX
            else:
                if not (self.includeLB_x and self.includeUB_x):
                    warn(
                        "General PWL functions must include both lower and upper bounds on the x variable."
                    )
                    return PWL1DType.UNABLE_TO_DETERMINE
                return PWL1DType.GENERAL
        elif is_concave and not is_convex:
            # Purely concave function
            if not self.includeUB_y:
                warn("Concave PWL functions must include an upper bound.")
                return PWL1DType.UNABLE_TO_DETERMINE
            if not self.includeLB_y:
                return PWL1DType.CONCAVE
            else:
                if not (self.includeLB_x and self.includeUB_x):
                    warn(
                        "General PWL functions must include both lower and upper bounds on the x variable."
                    )
                    return PWL1DType.UNABLE_TO_DETERMINE
                return PWL1DType.GENERAL
        else:
            # Both convex and concave - must be linear
            return PWL1DType.LINEAR

    def Plot(self, ax=None):
        """
        Create a visual plot of the PWL function.

        Generates a matplotlib plot showing the piecewise linear function with
        points marked and line segments connecting them. The plot includes
        the automatically determined function type in the title.

        Args:
            ax (matplotlib.axes.Axes, optional): Existing axes to plot on.
                If None, creates a new figure and axes.

        Returns:
            None: Displays the plot using plt.show().

        Example:
            >>> params = PWL1DParameters([(0, 1), (1, 3), (2, 2)])
            >>> params.Plot()  # Shows interactive plot

        Note:
            Requires matplotlib to be installed. The plot shows:
            - Connected line segments representing the PWL function
            - Circular markers at each breakpoint
            - Grid for easier reading
            - Function type in the title
        """
        import matplotlib.pyplot as plt

        # Extract x and y coordinates from points
        x_vals, y_vals = zip(*self.points)

        # Create new figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots()

        # Plot the PWL function with markers at breakpoints
        ax.plot(x_vals, y_vals, marker="o")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Piecewise Linear Function\n Type: {self.pwl_type.name}")
        ax.grid(True)
        plt.show()

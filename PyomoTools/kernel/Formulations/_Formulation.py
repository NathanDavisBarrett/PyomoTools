import pyomo.kernel as pmo

from abc import ABC
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Union, List, Dict, Tuple, Any
from itertools import product


class _Formulation(pmo.block, ABC):
    """
    A base class for all formulations in PyomoTools.
    This class inherits from pmo.block and can be extended to create specific formulations.
    """

    def __init__(
        self,
        variableNames: List[str],
        variableInfo: Dict[
            str, Tuple[Union[pmo.variable, pmo.expression], Tuple[float, float]]
        ],
    ):
        super().__init__()

        self.variableNames = variableNames
        self.originalVariables = [variableInfo[n][0] for n in variableNames]

        self.Setup()

        self.bounds = {}
        self.plotAvailable = True
        for i, name in enumerate(self.variableNames):
            bounds = variableInfo[name][1]
            if bounds is None:
                bounds = [None, None]
            bounds = list(bounds)
            if len(bounds) != 2:
                raise ValueError(
                    f"Bounds for variable '{name}' must be a tuple of length 2."
                )
            if bounds[0] is None:
                if (
                    hasattr(self.originalVariables[i], "lb")
                    and self.originalVariables[i].lb is not None
                ):
                    bounds[0] = self.originalVariables[i].lb
            if bounds[1] is None:
                if (
                    hasattr(self.originalVariables[i], "ub")
                    and self.originalVariables[i].ub is not None
                ):
                    bounds[1] = self.originalVariables[i].ub

            if bounds[0] is None or bounds[1] is None:
                self.plotAvailable = False
            self.bounds[name] = bounds

        self.constraintFunctions = []
        self.constraintNames = []

    def Setup(self):
        """
        A function to perform any setup needed before variable registration.
        """
        pass

    def registerConstraint(
        self, func: Callable[[Any], Union[pmo.expression, bool]], name: str = None
    ):
        """
        A method to register a constraint function with the formulation.
        Parameters
        ----------
        func: Callable[[list], Union[pmo.expression, bool]]
            A function that takes a list of variable values (matching the order they were registered) and returns a Pyomo expression or a boolean indicating feasibility.
        name: str
            The name of the constraint to be registered. If None, a default name will be generated.
        """
        self.constraintFunctions.append(func)

        # Actually make a constraint out of the function
        if name is None:
            name = f"Constraint_{len(self.constraintFunctions)}"

        self.constraintNames.append(name)

        expr = func(*self.originalVariables)
        if isinstance(expr, (bool, np.bool_)):
            return
        else:
            setattr(self, name, pmo.constraint(expr))

    def generateStandaloneModel(self) -> pmo.block:
        model = pmo.block()
        for name in self.variableNames:
            setattr(
                model,
                name,
                pmo.variable(lb=self.bounds[name][0], ub=self.bounds[name][1]),
            )

        for i in range(len(self.constraintFunctions)):
            func = self.constraintFunctions[i]
            expr = func(*[getattr(model, name) for name in self.variableNames])
            if isinstance(expr, (bool, np.bool_)):
                continue
            else:
                setattr(model, self.constraintNames[i], pmo.constraint(expr))
        return model

    def _testFeasibility(self, point: NDArray[np.float64]) -> bool:
        """
        A method to test the feasibility of a given point in the formulation.

        Parameters
        ----------
        point: NDArray[np.float64]
            A numpy array representing the point to test for feasibility. Dimensionality must match the number of variables in the formulation.
        """
        for func in self.constraintFunctions:
            result = func(*point)
            if not isinstance(result, (bool, np.bool_)):
                result = bool(int(round(pmo.value(result))))
            if not result:
                return False
        return True

    def Plot_Scatter(self, numSamples=10000, color="blue", dotSize=10):
        """
        A method to plot the formulation. Plotting is handled by randomly sampling combinations of variables, testing their feasibility, and plotting the feasible points.

        Parameters
        ----------
        numSamples: int, optional
            The number of random samples to generate for plotting. Default is 1000.
        color: str, optional
            The color of the points in the plot. Default is 'blue'.
        dotSize: int, optional
            The size of the points in the plot. Default is 10.
        """
        import matplotlib.pyplot as plt

        fig = plt.figure()
        if len(self.variableNames) == 2:
            ax = fig.add_subplot(111)
        elif len(self.variableNames) == 3:
            ax = fig.add_subplot(111, projection="3d")

        variableSamples = np.random.uniform(
            low=[self.bounds[var][0] for var in self.variableNames],
            high=[self.bounds[var][1] for var in self.variableNames],
            size=(numSamples, len(self.variableNames)),
        )
        samplesPer = int(numSamples ** (1 / (len(self.variableNames))))
        if samplesPer < 1:
            variableSamples = np.random.uniform(
                low=[self.bounds[var][0] for var in self.variableNames],
                high=[self.bounds[var][1] for var in self.variableNames],
                size=(numSamples, len(self.variableNames)),
            )
        else:
            sampleRanges = [
                np.linspace(self.bounds[var][0], self.bounds[var][1], samplesPer)
                for i, var in enumerate(self.variableNames)
            ]
            variableSamples = np.array(list(product(*sampleRanges)))

        feasibleMask = np.array(
            [self._testFeasibility(point) for point in variableSamples]
        )

        feasiblePoints = variableSamples[feasibleMask]
        if len(self.variableNames) in [0, 1]:
            raise NotImplementedError(
                "Plotting is not available for this formulation due to insufficient number of variables."
            )
        if len(self.variableNames) == 2:
            ax.scatter(
                feasiblePoints[:, 0], feasiblePoints[:, 1], s=dotSize, color=color
            )
            ax.set_xlabel(self.variableNames[0])
            ax.set_ylabel(self.variableNames[1])
        elif len(self.variableNames) == 3:
            ax.scatter(
                feasiblePoints[:, 0],
                feasiblePoints[:, 1],
                feasiblePoints[:, 2],
                s=dotSize,
                color=color,
            )
            ax.set_xlabel(self.variableNames[0])
            ax.set_ylabel(self.variableNames[1])
            ax.set_zlabel(self.variableNames[2])
        else:
            # Plot the first three variables and leave the remainder as sliders
            from matplotlib.widgets import Slider

            # Create 3D subplot for the first three variables
            ax = fig.add_subplot(111, projection="3d")

            # Initial values for slider variables (midpoint of their bounds)
            initial_slider_values = []
            for i in range(3, len(self.variableNames)):
                var_name = self.variableNames[i]
                low, high = self.bounds[var_name]
                initial_slider_values.append((low + high) / 2)

            def update_plot(slider_values):
                """Update the 3D plot based on current slider values."""
                ax.clear()

                # Filter feasible points based on slider constraints
                # For higher dimensions, we only show points where the slider variables
                # are within a small tolerance of the slider values
                tolerance = 0.05  # 5% tolerance
                filtered_points = []

                for point in feasiblePoints:
                    # Check if this point's slider dimensions match current slider values
                    matches_sliders = True
                    for i, slider_val in enumerate(slider_values):
                        dim_idx = i + 3  # slider dimensions start after first 3
                        var_name = self.variableNames[dim_idx]
                        var_range = self.bounds[var_name][1] - self.bounds[var_name][0]
                        if abs(point[dim_idx] - slider_val) > tolerance * var_range:
                            matches_sliders = False
                            break

                    if matches_sliders:
                        filtered_points.append(
                            point[:3]
                        )  # Only first 3 dimensions for plotting

                if len(filtered_points) > 0:
                    filtered_points = np.array(filtered_points)
                    ax.scatter(
                        filtered_points[:, 0],
                        filtered_points[:, 1],
                        filtered_points[:, 2],
                        s=dotSize,
                        color=color,
                    )

                ax.set_xlabel(self.variableNames[0])
                ax.set_ylabel(self.variableNames[1])
                ax.set_zlabel(self.variableNames[2])

                # Add title showing current slider values
                slider_info = ", ".join(
                    [
                        f"{self.variableNames[i+3]}={val:.2f}"
                        for i, val in enumerate(slider_values)
                    ]
                )
                ax.set_title(f"Feasible Region ({slider_info})")

                plt.draw()

            # Create sliders for dimensions 4 and beyond
            slider_axes = []
            sliders = []

            # Adjust figure to make room for sliders
            plt.subplots_adjust(bottom=0.1 + 0.05 * len(initial_slider_values))

            for i, slider_val in enumerate(initial_slider_values):
                var_name = self.variableNames[i + 3]
                low, high = self.bounds[var_name]

                # Create axis for this slider
                slider_ax = plt.axes([0.2, 0.05 + i * 0.04, 0.5, 0.03])
                slider_axes.append(slider_ax)

                # Create slider
                if samplesPer < 1:
                    slider = Slider(slider_ax, var_name, low, high, valinit=slider_val)
                else:
                    slider = Slider(
                        slider_ax,
                        var_name,
                        low,
                        high,
                        valinit=slider_val,
                        valstep=sampleRanges[i + 3],
                    )
                sliders.append(slider)

            def on_slider_change(val):
                """Callback function for when any slider changes."""
                current_values = [s.val for s in sliders]
                update_plot(current_values)

            # Connect all sliders to the update function
            for slider in sliders:
                slider.on_changed(on_slider_change)

            # Initial plot
            update_plot(initial_slider_values)

        plt.show()

    def Plot(self):
        standaloneModel = self.generateStandaloneModel()

        from ..MatrixRepresentation import MatrixRepresentation

        mr = MatrixRepresentation(standaloneModel)
        mr.Plot()

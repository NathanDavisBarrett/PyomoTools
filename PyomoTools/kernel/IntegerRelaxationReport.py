import pyomo.kernel as pmo
import warnings
from functools import cached_property

from .RelaxIntegerVarsKernel import RelaxIntegerVarsKernel as RelaxIntegerVars
from .ParallelComponentIterator import ParallelComponentIterator
from ..base.Solvers import DefaultSolver


class IntegerRelaxationReport:
    """
    A class to generate a report on the integer relaxation of a Pyomo model.

    Upon construction, this class solves the integer relaxation of the provided Pyomo model using the specified solver (or a default LP solver if none is provided).
    It then collects the following information:
    - fractionality_ratio: The ratio of the number of non-integral integer values in the LP relaxation to the total number of integer variables.
    - sum_integer_infeasibility: The sum of the absolute values of the deviations of integer variables from their nearest integer values in the LP relaxation.
    - max_integer_infeasibility: The maximum absolute deviation of any integer variable from its nearest integer value in the LP relaxation.
    - euclidean_distance_to_nearest_neighbor: The Euclidean distance from the LP relaxation solution to the nearest (i.e. rounded) integer solution (not necessarily feasible).
    - gap_to_integer_solution: The gap between the objective value of the LP relaxation and the solution contained withing the model upon construction of this report.

    NOTE: The LP-relaxed model is stored in the `lp_relaxation` attribute.
    """

    def __init__(
        self,
        model: pmo.block,
        solver=None,
        tee: bool = False,
        solver_options={},
        tolerance: float = 1e-5,
    ):
        """
        Initialize the IntegerRelaxationReport with a Pyomo model and a solver.

        Parameters
        ----------
        model : pmo.block
            The Pyomo model to analyze (must be solved already)
        solver : pyomo solver object, optional
            The solver to use for solving the integer relaxation. If None, the default LP solver is used (from PyomoTools.base.Solvers.DefaultSolver).
        tee : bool, optional
            Whether to display solver output during solving.
        solver_options : dict, optional
            Additional options to pass to the solver upon solving.
        tolerance : float, optional
            The tolerance for determining integrality of variables. Defaults to 1e-5.
        """
        self.model = model
        self.tolerance = tolerance
        self.solver = solver if solver is not None else DefaultSolver("LP")

        self.lp_relaxation = self.model.clone()
        RelaxIntegerVars().apply_to(self.lp_relaxation)
        results = self.solver.solve(
            self.lp_relaxation, tee=False, options=solver_options
        )
        if results.solver.termination_condition != pmo.TerminationCondition.optimal:
            warnings.warn(
                "IntegerRelaxationReport: The LP relaxation did not solve to optimality."
            )

    @cached_property
    def fractionality_ratio(self) -> float:
        numNonIntegral = 0
        totalIntegerVars = 0
        for originalVar, relaxedVar in ParallelComponentIterator(
            [self.model, self.lp_relaxation],
            collect_vars=True,
            collect_constrs=False,
            collect_objs=False,
        ):
            if originalVar.is_integer():
                totalIntegerVars += 1
                val = pmo.value(relaxedVar)
                if abs(val - round(val)) > self.tolerance:
                    numNonIntegral += 1
        if totalIntegerVars == 0:
            return 0.0
        return numNonIntegral / totalIntegerVars

    @cached_property
    def _integer_infeasibilities(self):
        sum_infeasibility = 0.0
        max_infeasibility = 0.0
        euclidean_distance_squared = 0.0

        for originalVar, relaxedVar in ParallelComponentIterator(
            [self.model, self.lp_relaxation],
            collect_vars=True,
            collect_constrs=False,
            collect_objs=False,
        ):
            if originalVar.is_integer():
                val = pmo.value(relaxedVar)
                infeasibility = abs(val - round(val))
                sum_infeasibility += infeasibility
                euclidean_distance_squared += infeasibility**2
                if infeasibility > max_infeasibility:
                    max_infeasibility = infeasibility
        return sum_infeasibility, max_infeasibility, euclidean_distance_squared

    @cached_property
    def sum_integer_infeasibility(self) -> float:
        return self._integer_infeasibilities[0]

    @cached_property
    def max_integer_infeasibility(self) -> float:
        return self._integer_infeasibilities[1]

    @cached_property
    def euclidean_distance_to_nearest_neighbor(self) -> float:
        return self._integer_infeasibilities[2] ** 0.5

    @cached_property
    def gap_to_integer_solution(self) -> float:
        for milp_obj, lp_obj in ParallelComponentIterator(
            [self.model, self.lp_relaxation],
            collect_vars=False,
            collect_constrs=False,
            collect_objs=True,
        ):
            # Assuming the first objective detected is the main one
            milp_obj_value = pmo.value(milp_obj)
            lp_obj_value = pmo.value(lp_obj)
            if abs(milp_obj_value) < 1e-9:
                return abs(
                    lp_obj_value - milp_obj_value
                )  # Return absolute error if obj is 0
            return abs(lp_obj_value - milp_obj_value) / abs(milp_obj_value)

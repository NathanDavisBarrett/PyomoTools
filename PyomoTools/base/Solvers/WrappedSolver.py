import pyomo.kernel as pmo

from ...kernel.FindLeastInfeasibleSolution import (
    FindLeastInfeasibleSolution,
    LeastInfeasibleDefinition,
)
from ...kernel.InfeasibilityReport import InfeasibilityReport
from ...kernel.IO import ModelToJson

import warnings


class WrappedSolver:
    def __init__(
        self,
        solver,
        leastInfeasibleDefinition: LeastInfeasibleDefinition = LeastInfeasibleDefinition.L1_Norm,
        infeasibilityReportFileName: str = "infeasibilityReport.txt",
        solutionJsonFileName: str = "leastInfeasibleSolution.json",
        exception: bool = True,
        defaultSolverOptions={},
        infeasibilityReportKwargs={},
    ):
        self.solver = solver
        self.leastInfeasibleDefinition = leastInfeasibleDefinition
        self.infeasibilityReportFileName = infeasibilityReportFileName
        self.solutionJsonFileName = solutionJsonFileName
        self.exception = exception
        for k in defaultSolverOptions:
            self.solver.options[k] = defaultSolverOptions[k]
        self.infeasibilityReportKwargs = infeasibilityReportKwargs

    def solve(self, model, *args, **kwargs):
        result = self.solver.solve(model, *args, **kwargs)
        if result.solver.termination_condition in [
            pmo.TerminationCondition.infeasible,
            pmo.TerminationCondition.infeasibleOrUnbounded,
        ]:
            warnings.warn(
                "The model was infeasible. Attempting to find a least infeasible solution."
            )
            FindLeastInfeasibleSolution(
                model,
                self.solver,
                leastInfeasibleDefinition=self.leastInfeasibleDefinition,
                solver_args=args,
                solver_kwargs=kwargs,
            )
            if self.infeasibilityReportFileName is not None:
                report = InfeasibilityReport(model, **self.infeasibilityReportKwargs)
                report.WriteFile(self.infeasibilityReportFileName)
                repMessage = f"The infeasibility report for a least-infeasible solution was written to {self.infeasibilityReportFileName}.\n"
            else:
                repMessage = ""

            if self.solutionJsonFileName is not None:
                ModelToJson(model, self.solutionJsonFileName)
                solMessage = f"The least infeasible solution was written to {self.solutionJsonFileName}.\n"
            else:
                solMessage = ""

            message = f"The model was infeasible.\n{repMessage}{solMessage}"
            if self.exception:
                raise ValueError(message)
            else:
                warnings.warn(message)

        return result

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
        interactiveInfeasibilityReport: bool = False,
        solutionJsonFileName: str = "leastInfeasibleSolution.json",
        exception: bool = True,
        warn: bool = True,
        defaultSolverOptions={},
        infeasibilityReportKwargs={},
    ):
        self.solver = solver
        self.leastInfeasibleDefinition = leastInfeasibleDefinition
        self.infeasibilityReportFileName = infeasibilityReportFileName
        self.interactiveInfeasibilityReport = interactiveInfeasibilityReport
        self.solutionJsonFileName = solutionJsonFileName
        self.exception = exception
        self.warn = warn
        for k in defaultSolverOptions:
            self.solver.options[k] = defaultSolverOptions[k]
        self.infeasibilityReportKwargs = infeasibilityReportKwargs

    def solve(
        self,
        model,
        *args,
        relax_only_these_constraints: list = None,
        retry_original_objective=False,
        exception: bool = None,
        warn: bool = True,
        **kwargs,
    ):
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
                relax_only_these_constraints=relax_only_these_constraints,
                retry_original_objective=retry_original_objective,
            )
            if self.solutionJsonFileName is not None:
                ModelToJson(model, self.solutionJsonFileName)
                solMessage = f"The least infeasible solution was written to {self.solutionJsonFileName}.\n"
            else:
                solMessage = ""

            if self.infeasibilityReportFileName is not None:
                if self.interactiveInfeasibilityReport:
                    from ...kernel.InfeasibilityReport_Interactive import (
                        InfeasibilityReport_Interactive,
                    )

                    rep = InfeasibilityReport_Interactive(
                        model, **self.infeasibilityReportKwargs
                    )
                    rep.show()
                    repMessage = "The interactive infeasibility report was shown.\n"
                else:
                    report = InfeasibilityReport(
                        model, **self.infeasibilityReportKwargs
                    )
                    report.WriteFile(self.infeasibilityReportFileName)
                    repMessage = f"The infeasibility report for a least-infeasible solution was written to {self.infeasibilityReportFileName}.\n"
            else:
                repMessage = ""

            message = f"The model was infeasible.\n{repMessage}{solMessage}"
            if exception is None:
                exception = self.exception
            if exception:
                raise ValueError(message)

            if warn is None:
                warn = self.warn
            if warn:
                warnings.warn(message)

        return result

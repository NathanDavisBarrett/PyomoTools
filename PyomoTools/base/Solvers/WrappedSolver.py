import pyomo.kernel as pmo

from ...kernel.FindLeastInfeasibleSolution import FindLeastInfeasibleSolution, LeastInfeasibleDefinition
from ...kernel.InfeasibilityReport import InfeasibilityReport

import warnings

class WrappedSolver:
    def __init__(
            self, 
            solver,
            leastInfeasibleDefinition:LeastInfeasibleDefinition=LeastInfeasibleDefinition.L1_Norm,
            infeasibilityReportFileName:str="infeasibilityReport.txt",
            exception:bool=True,
            defaultSolverOptions={},
            infeasibilityReportKwargs={}
            ):
        self.solver = solver
        self.leastInfeasibleDefinition = leastInfeasibleDefinition
        self.infeasibilityReportFileName = infeasibilityReportFileName
        self.exception = exception
        for k in defaultSolverOptions:
            self.solver.options[k] = defaultSolverOptions[k]
        self.infeasibilityReportKwargs = infeasibilityReportKwargs



    def solve(self,model,*args,**kwargs):
        result = self.solver.solve(model,*args,**kwargs)
        if result.solver.termination_condition == pmo.TerminationCondition.infeasible:
            warnings.warn("The model was infeasible. Attempting to find a least infeasible solution.")
            FindLeastInfeasibleSolution(model,self.solver,leastInfeasibleDefinition=self.leastInfeasibleDefinition,solver_args=args,solver_kwargs=kwargs)
            report = InfeasibilityReport(model,**self.infeasibilityReportKwargs)
            report.WriteFile(self.infeasibilityReportFileName)
            message = f"The model was infeasible. The infeasibility report for a least-infeasible solution was written to {self.infeasibilityReportFileName}."
            if self.exception:
                raise ValueError(message)
            else:
                warnings.warn(message)

        return result


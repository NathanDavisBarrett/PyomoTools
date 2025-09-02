from pyomo.opt import SolverFactory
import time
import gurobipy as gp


class TimedIncumbentSolver:
    # Custom callback for termination logic
    class TimedIncumbentCallback:
        """
        A callback class to terminate the optimization process after a specified time limit only if an incumbent solution is found, otherwise continue until an incumbent is found.
        """

        def __init__(self, timeLimit=60):
            self.timeLimit = timeLimit
            self.tic()

        def __call__(self, _, solver, where):
            if where == gp.GRB.Callback.MIP:
                solcount = solver._solver_model.cbGet(gp.GRB.Callback.MIP_SOLCNT)

                if solcount > 0 and time.time() - self.start_time >= self.timeLimit:
                    solver._solver_model.terminate()

        def tic(self):
            self.start_time = time.time()

    def __init__(self, timeLimit=60):
        self.timeLimit = timeLimit
        self.solver = SolverFactory("gurobi_persistent")
        self.cb = TimedIncumbentSolver.TimedIncumbentCallback(timeLimit)

        self.solver.set_callback(self.cb)

    def solve(self, model, *args, **kwargs):
        """
        Solve the given Pyomo model using the Gurobi persistent solver with a timed incumbent callback.

        Parameters:
        model: pyomo.environ.ConcreteModel
            The Pyomo model to solve.
        """
        self.cb.tic()
        self.solver.set_instance(model)
        return self.solver.solve(model, *args, **kwargs)


# import pyomo.environ as pyo
# from pyomo.opt import SolverFactory
# import time
# import gurobipy as gp

# class TimedIncumbentSolver:
#     # Custom callback for termination logic
#     class TimedIncumbentCallback:
#         """
#         A callback class to terminate the optimization process after a specified time limit only if an incumbent solution is found, otherwise continue until an incumbent is found.
#         """
#         def __init__(self,timeLimit=60):
#             self.start_time = time.time()
#             self.timeLimit = timeLimit
#             self.incumbent_found = False

#         def __call__(self, model, where):
#             if where == gp.GRB.Callback.MIPSOL:
#                 self.incumbent_found = True

#             if self.incumbent_found and time.time() - self.start_time >= self.timeLimit:
#                 model.terminate()


#     def __init__(self, timeLimit=60):
#         self.timeLimit = timeLimit
#         self.solver = SolverFactory('gurobi_persistent')
#         self.cb = TimedIncumbentSolver.TimedIncumbentCallback(timeLimit)

#     def solve(self, model, tee=False, options={}):
#         """
#         Solve the given Pyomo model using the Gurobi persistent solver with a timed incumbent callback.

#         Parameters:
#         model: pyomo.environ.ConcreteModel
#             The Pyomo model to solve.
#         tee: bool (optional, Default=False)
#             Whether to print solver output to the console.
#         options: dict (optional, Default={})
#             A dictionary of solver options.
#         """
#         self.solver.set_instance(model)

#         if tee:
#             self.solver._solver_model.setParam('OutputFlag', 1)
#         else:
#             self.solver._solver_model.setParam('OutputFlag', 0)

#         for option, value in options.items():
#             self.solver._solver_model.setParam(option, value)

#         self.solver._solver_model.optimize(self.cb)
#         self.solver.load_vars()

#         results = self.solver._postsolve()
#         return results

import pyomo.environ as pyo
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
from pyomo.core.expr.visitor import identify_variables

from gurobipy import GRB

from typing import List, Union, Callable
from collections import deque


class GurobiPersistent_WithImplicitLazyConstraints(GurobiPersistent):
    """
    A Gurobi persistent solver that supports implicit lazy constraints.

    With this approach, constraints are generated dynamically via a callback
    function during the solve process. When a MIP solution is found that
    violates implicit constraints, those constraints are added as lazy constraints.
    """

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.set_callback(self._gurobi_callback)
        self.n_calls = 0

    def _gurobi_callback(self, model, solver, where):
        """Gurobi callback for handling lazy constraints."""
        if where == GRB.Callback.MIPSOL:
            self.n_calls += 1
            # Get the current solution
            solver.cbGetSolution(self.relevant_pyomo_variables)

            # Generate new constraints
            new_constraints = self.constraint_generator(self._pyomo_model)
            for con_expr in new_constraints:
                # Add each new constraint as a lazy constraint
                tmp_c = pyo.Constraint(expr=con_expr)
                tmp_c.construct()
                solver.cbLazy(tmp_c)

    def set_instance(
        self,
        model: pyo.ConcreteModel,
        constraint_generator: Callable[[pyo.ConcreteModel], List],
        relevant_components: List = None,
        **kwds,
    ):
        """
        Set the model instance and configure implicit lazy constraints.

        Parameters
        ----------
        model : pyo.ConcreteModel
            The Pyomo model to solve
        constraint_generator : Callable
            A function that takes the model and returns a list of constraint
            expressions that are currently violated
        relevant_components : list, optional
            List of model components that are relevant for the callback.
            If None, all variables in the model are considered relevant.
        """
        super().set_instance(model, **kwds)
        self._pyomo_model = model
        self._solver_model.update()
        self.n_calls = 0

        # Gurobi only needs relevant variable values in the callback
        if relevant_components is None:
            # By default, use all variables in the model
            relevant_components = list(model.component_objects(pyo.Var, active=True))

        self.relevant_gurobi_variables = []
        self.relevant_pyomo_variables = []

        # Build mapping of variables
        seen_vars = set()
        for comp in relevant_components:
            if isinstance(comp, pyo.Var):
                for var_data in comp.values():
                    if id(var_data) not in seen_vars:
                        seen_vars.add(id(var_data))
                        gurobi_var = self._pyomo_var_to_solver_var_map.get(var_data)
                        if gurobi_var is not None:
                            self.relevant_gurobi_variables.append(gurobi_var)
                            self.relevant_pyomo_variables.append(var_data)
            elif isinstance(comp, pyo.Constraint):
                for con_data in comp.values():
                    for var in identify_variables(con_data.expr):
                        if id(var) not in seen_vars:
                            seen_vars.add(id(var))
                            gurobi_var = self._pyomo_var_to_solver_var_map.get(var)
                            if gurobi_var is not None:
                                self.relevant_gurobi_variables.append(gurobi_var)
                                self.relevant_pyomo_variables.append(var)

        self.constraint_generator = constraint_generator

    def solve(self, *args, **kwds):
        """Solve with implicit lazy constraints enabled."""
        if "options" not in kwds:
            kwds["options"] = {}
        if "LazyConstraints" not in kwds["options"]:
            kwds["options"]["LazyConstraints"] = 1
        return super().solve(*args, **kwds)

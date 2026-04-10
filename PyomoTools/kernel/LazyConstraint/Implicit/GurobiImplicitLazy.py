import pyomo.kernel as pmo
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
from pyomo.core.expr.visitor import identify_variables

from gurobipy import GRB

from typing import List, Union, Callable
from collections import deque  # For fast appends


class GurobiPersistent_WithImplicitLazyConstraints(GurobiPersistent):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.set_callback(self._gurobi_callback)
        self.n_calls = 0

    def _gurobi_callback(self, model: pmo.block, solver: GurobiPersistent, where):
        self.n_calls += 1
        if where == GRB.Callback.MIPSOL:
            solver.cbGetSolution(self.relevant_pyomo_variables)

            new_constraints = self.constraint_generator(model)
            for con in new_constraints:
                solver.cbLazy(pmo.constraint(con))

    def set_instance(
        self,
        model: pmo.block,
        constraint_generator: Callable[[pmo.block], List[pmo.expression]],
        relevant_components: List[
            Union[
                pmo.variable,
                pmo.variable_list,
                pmo.variable_tuple,
                pmo.variable_dict,
                pmo.constraint,
                pmo.constraint_list,
                pmo.constraint_tuple,
                pmo.constraint_dict,
                pmo.block,
                pmo.block_list,
                pmo.block_tuple,
                pmo.block_dict,
            ]
        ] = None,
        **kwds,
    ):
        super().set_instance(model, **kwds)
        self._solver_model.update()
        self.n_calls = 0

        # Gurobi will only need to load relevant variable values in the callback.
        # A variable will be relevant if it is explicitly included, or mentioned in a constraint or block that is included.
        if relevant_components is None:
            relevant_components = [
                model
            ]  # By default, we consider all components relevant

        # Gurobi variables are hashable, pyomo variables are not.
        # So we first create a unique set of gurobi variables and then map back to pyomo variables.
        self.relevant_gurobi_variables = deque()
        self.add_relevant_variables(relevant_components)
        self.relevant_gurobi_variables = list(set(self.relevant_gurobi_variables))
        self.relevant_pyomo_variables = [
            self._solver_var_to_pyomo_var_map.get(var)
            for var in self.relevant_gurobi_variables
        ]

        self.constraint_generator = constraint_generator

    def add_relevant_variables_from_constraint(self, constraint):
        for var in identify_variables(constraint.body):
            self.relevant_gurobi_variables.append(
                self._pyomo_var_to_solver_var_map.get(var)
            )

    def add_relevant_variables(self, relevant_components):
        for comp in relevant_components:
            if isinstance(comp, pmo.variable):
                self.relevant_gurobi_variables.append(
                    self._pyomo_var_to_solver_var_map.get(comp)
                )
            elif isinstance(comp, (pmo.variable_list, pmo.variable_tuple)):
                for var in comp:
                    self.relevant_gurobi_variables.append(
                        self._pyomo_var_to_solver_var_map.get(var)
                    )
            elif isinstance(comp, pmo.variable_dict):
                for var in comp.values():
                    self.relevant_gurobi_variables.append(
                        self._pyomo_var_to_solver_var_map.get(var)
                    )
            elif isinstance(comp, pmo.constraint):
                self.add_relevant_variables_from_constraint(comp)
            elif isinstance(comp, (pmo.constraint_list, pmo.constraint_tuple)):
                for con in comp:
                    self.add_relevant_variables_from_constraint(con)
            elif isinstance(comp, pmo.constraint_dict):
                for con in comp.values():
                    self.add_relevant_variables_from_constraint(con)
            elif isinstance(comp, pmo.block):
                for child in comp.children():
                    self.add_relevant_variables([child])
            elif isinstance(comp, (pmo.block_list, pmo.block_tuple)):
                for b in comp:
                    self.add_relevant_variables([b])
            elif isinstance(comp, pmo.block_dict):
                for b in comp.values():
                    self.add_relevant_variables([b])

    def solve(self, *args, **kwds):
        if "options" not in kwds:
            kwds["options"] = {}
        if "LazyConstraints" not in kwds["options"]:
            kwds["options"]["LazyConstraints"] = 1
        return super().solve(*args, **kwds)

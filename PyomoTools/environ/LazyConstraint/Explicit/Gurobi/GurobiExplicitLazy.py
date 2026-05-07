import pyomo.environ as pyo
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent


class GurobiPersistent_WithExplicitLazyConstraints(GurobiPersistent):
    """
    A Gurobi persistent solver that supports explicit lazy constraints.

    Lazy constraints are explicitly provided to the set_instance method
    and are assigned the Lazy=1 attribute in Gurobi.
    """

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def _label_lazy_constraint(self, con):
        """Mark a single lazy constraint in Gurobi."""
        gurobi_c = self._pyomo_con_to_solver_con_map.get(con)
        if gurobi_c is not None:
            gurobi_c.Lazy = 1

    def _label_lazy_constraints(self, lazy_constraints):
        """Find and mark all explicitly provided lazy constraints."""
        if lazy_constraints is None:
            return
        # Wrap in a list if it's a single component so we can iterate
        if not isinstance(lazy_constraints, list):
            lazy_constraints = [lazy_constraints]

        for con in lazy_constraints:
            # Check if it's an IndexedComponent (has values() method)
            if hasattr(con, "values"):
                for con_data in con.values():
                    self._label_lazy_constraint(con_data)
            else:
                # It's a SimpleConstraint or a single ConstrData
                self._label_lazy_constraint(con)

    def set_instance(self, model, lazy_constraints=None, **kwds):
        """Set the model instance and mark lazy constraints."""
        super().set_instance(model, **kwds)

        # Find and label lazy constraints
        self._label_lazy_constraints(lazy_constraints)

        # Update the solver model
        self._solver_model.update()

    def solve(self, *args, **kwds):
        """Solve with lazy constraints enabled."""
        if "options" not in kwds:
            kwds["options"] = {}
        if "LazyConstraints" not in kwds["options"]:
            kwds["options"]["LazyConstraints"] = 1
        return super().solve(*args, **kwds)

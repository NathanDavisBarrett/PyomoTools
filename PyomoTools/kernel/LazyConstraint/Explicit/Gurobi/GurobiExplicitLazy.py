import pyomo.kernel as pmo
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent

from ..LazyConstraint import lazy_constraint


class GurobiPersistent_WithExplicitLazyConstraints(GurobiPersistent):
    def __init__(self, **kwds):
        super().__init__(**kwds)

    def _label_lazy_constraint(self, c: lazy_constraint):
        gurobi_c = self._pyomo_con_to_solver_con_map.get(c)
        if gurobi_c is not None:
            gurobi_c.Lazy = 1

    def _label_lazy_constraints(self, model: pmo.block):
        for child in model.children():
            if isinstance(child, lazy_constraint) and child.active:
                self._label_lazy_constraint(child)
            elif isinstance(child, (pmo.constraint_list, pmo.constraint_tuple)):
                for c in child:
                    if isinstance(c, lazy_constraint) and c.active:
                        self._label_lazy_constraint(c)
            elif isinstance(child, pmo.constraint_dict):
                for c in child.values():
                    if isinstance(c, lazy_constraint) and c.active:
                        self._label_lazy_constraint(c)
            elif isinstance(child, pmo.block) and child.active:
                self._label_lazy_constraints(child)
            elif isinstance(child, (pmo.block_list, pmo.block_tuple)):
                for b in child:
                    self._label_lazy_constraints(b)
            elif isinstance(child, pmo.block_dict):
                for b in child.values():
                    self._label_lazy_constraints(b)

    def set_instance(self, model, **kwds):
        super().set_instance(model, **kwds)

        # Track down lazy constraints and label their gurobi counterparts
        self._label_lazy_constraints(model)

        self._solver_model.update()

    def solve(self, *args, **kwds):
        if "options" not in kwds:
            kwds["options"] = {}
        if "LazyConstraints" not in kwds["options"]:
            kwds["options"]["LazyConstraints"] = 1
        return super().solve(*args, **kwds)

import pyomo.kernel as pmo
from pyomo.core import Transformation


class RelaxIntegerVarsKernel(Transformation):
    """
    A transformation that relaxes the integrality constraints of
    integer and binary variables in a Pyomo model.

    NOTE: Pyomo has a default implementation of this transformation (core.relax_integer_vars) but it appears to only be compatible with the pyomo environment, not the pyomo kernel.
    """

    def _relax_integer_var(self, var: pmo.variable):
        if var.is_integer():
            lb = var.lb
            ub = var.ub
            var.domain = pmo.Reals
            var.lb = lb
            var.ub = ub

    def _apply_to(self, instance, **kwds):
        for e in instance.children():
            if isinstance(e, pmo.variable):
                self._relax_integer_var(e)
            elif isinstance(e, (pmo.variable_list, pmo.variable_tuple)):
                for var in e:
                    self._relax_integer_var(var)
            elif isinstance(e, pmo.variable_dict):
                for var in e.values():
                    self._relax_integer_var(var)
            elif isinstance(e, pmo.block):
                self._apply_to(e, **kwds)
            elif isinstance(e, (pmo.block_list, pmo.block_tuple)):
                for b in e:
                    self._apply_to(b, **kwds)
            elif isinstance(e, pmo.block_dict):
                for b in e.values():
                    self._apply_to(b, **kwds)

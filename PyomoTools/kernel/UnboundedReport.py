import pyomo.kernel as pmo


class UnboundedReport:
    """
    A class to report unbounded variables and constraints in a Pyomo model.

    This class collects information about unbounded variables and constraints
    during the optimization process and provides a summary report.

    Attributes
    ----------
    unbounded_vars : list
        A list to store unbounded variable names.
    """

    def __init__(
        self,
        model: pmo.block,
        solver: int = 1e10,
        unboundedLimit: float = 1e10,
        rTol: float = 1e-5,
    ):
        assert unboundedLimit > 0, "unboundedLimit must be a positive number."
        self.unbounded_vars = []

        self.model = model
        self.unboundedLimit = unboundedLimit
        self.rTol = rTol

        self.set_var_bounds(self.model)

        results = solver.solve(self.model, tee=False)
        assert (
            results.solver.termination_condition != pmo.TerminationCondition.unbounded
        ), "The model is unbounded even after setting variable bounds. Consider decreasing unboundedLimit."
        self.find_unbounded(self.model)

    def set_var_bounds(self, block: pmo.block):
        for child in block.children():
            if isinstance(child, pmo.variable):
                if child.lb is None:
                    child.lb = -self.unboundedLimit
                if child.ub is None:
                    child.ub = self.unboundedLimit
            elif isinstance(child, (pmo.variable_list, pmo.variable_tuple)):
                for var in child:
                    if var.lb is None:
                        var.lb = -self.unboundedLimit
                    if var.ub is None:
                        var.ub = self.unboundedLimit
            elif isinstance(child, pmo.variable_dict):
                for var in child.values():
                    if var.lb is None:
                        var.lb = -self.unboundedLimit
                    if var.ub is None:
                        var.ub = self.unboundedLimit
            elif isinstance(child, pmo.block):
                self.set_var_bounds(child)
            elif isinstance(child, (pmo.block_list, pmo.block_tuple)):
                for blk in child:
                    self.set_var_bounds(blk)
            elif isinstance(child, pmo.block_dict):
                for blk in child.values():
                    self.set_var_bounds(blk)

    def test_unbounded(self, var: pmo.variable):
        val = pmo.value(var, exception=False)
        if val is None:
            return
        if abs(val - self.unboundedLimit) / self.unboundedLimit >= self.rTol:
            self.unbounded_vars.append((var, 0))
        elif abs(val + self.unboundedLimit) / self.unboundedLimit >= self.rTol:
            self.unbounded_vars.append((var, 1))

    def find_unbounded(self, block: pmo.block):
        for child in block.children():
            if isinstance(child, pmo.variable):
                self.test_unbounded(child)
            elif isinstance(child, (pmo.variable_list, pmo.variable_tuple)):
                for var in child:
                    if var.lb is None or var.ub is None:
                        self.test_unbounded(var)
            elif isinstance(child, pmo.variable_dict):
                for var in child.values():
                    if var.lb is None or var.ub is None:
                        self.test_unbounded(var)
            elif isinstance(child, pmo.block):
                self.find_unbounded(child)
            elif isinstance(child, (pmo.block_list, pmo.block_tuple)):
                for blk in child:
                    self.find_unbounded(blk)
            elif isinstance(child, pmo.block_dict):
                for blk in child.values():
                    self.find_unbounded(blk)

    def __str__(self):
        report_lines = ["Unbounded Variables:"]
        for var in self.unbounded_vars:
            report_lines.append(
                f" - {var[0].name}: {'+' if var[1]==0 else '-'}infinity"
            )
        return "\n".join(report_lines)

    def WriteFile(self, filename: str):
        with open(filename, "w") as f:
            f.write(str(self))

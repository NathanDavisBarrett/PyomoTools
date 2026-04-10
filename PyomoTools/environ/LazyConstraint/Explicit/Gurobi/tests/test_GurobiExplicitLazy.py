import pyomo.environ as pyo
import numpy as np
from pyomo.common.tee import capture_output
import io
import re

from ..GurobiExplicitLazy import GurobiPersistent_WithExplicitLazyConstraints


def test_InfiniteOpt():
    """Test explicit lazy constraints with a circle constraint."""
    model = pyo.ConcreteModel()
    mag = 9

    model.x = pyo.Var(bounds=(-mag, mag), domain=pyo.Integers)
    model.y = pyo.Var(bounds=(-mag, mag), domain=pyo.Integers)
    model.obj = pyo.Objective(expr=model.x + model.y, sense=pyo.maximize)

    # X and Y are bound to within the unit circle using lazy constraints
    n_points = 10
    model.angles = pyo.Set(
        initialize=list(np.linspace(0, 2 * np.pi, n_points, endpoint=False))
    )

    def c_rule(m, angle):
        return m.x * np.cos(angle) + m.y * np.sin(angle) <= mag

    model.c = pyo.Constraint(model.angles, rule=c_rule)

    opt = GurobiPersistent_WithExplicitLazyConstraints()
    opt.set_instance(model, lazy_constraints=[model.c])

    # Deactivate presolve and heuristics so Gurobi doesn't ignore lazy constraints
    opt.options["Presolve"] = 0
    opt.options["Heuristics"] = 0

    stream = io.StringIO()
    with capture_output(stream):
        results = opt.solve(tee=True)

    log = stream.getvalue()

    assert results.solver.status == "ok"
    assert results.solver.termination_condition == "optimal"

    lazy_constraint_pattern = re.compile(r"Lazy constraints:\s+(\d+)")
    match = lazy_constraint_pattern.search(log)
    assert (
        match is not None
    ), "Could not find 'Lazy constraints' in the log which indicates that the lazy constraints were not processed."
    n_lazy_constraints = int(match.group(1))
    assert (
        n_lazy_constraints > 0
    ), "Expected at least one lazy constraint to be added, but found none."


if __name__ == "__main__":
    test_InfiniteOpt()
    print("Explicit lazy constraint test passed!")

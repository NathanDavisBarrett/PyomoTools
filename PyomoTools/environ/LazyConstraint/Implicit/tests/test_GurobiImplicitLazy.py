import pyomo.environ as pyo
import numpy as np
from pyomo.common.tee import capture_output
import io
import re

from ..GurobiImplicitLazy import GurobiPersistent_WithImplicitLazyConstraints


def test_InfiniteOpt():
    """Test implicit lazy constraints with dynamic constraint generation."""
    model = pyo.ConcreteModel()
    mag = 9
    generated_constraint_coeffs = []

    model.x = pyo.Var(bounds=(-mag, mag), domain=pyo.Integers)
    model.y = pyo.Var(bounds=(-mag, mag), domain=pyo.Integers)
    model.obj = pyo.Objective(expr=model.x + model.y, sense=pyo.maximize)

    def constraint_generator(m):
        """
        Generate lazy constraints to enforce the unit circle constraint.
        """
        x = pyo.value(m.x)
        y = pyo.value(m.y)

        mag_i = np.sqrt(x**2 + y**2)
        if mag_i <= mag + 1e-2:
            return []

        # Generate a tangent constraint at the current point
        theta = np.arctan2(y, x)
        a = float(np.cos(theta))
        b = float(np.sin(theta))
        generated_constraint_coeffs.append((a, b, mag))

        # Create constraint expression
        new_con = m.x * a + m.y * b <= mag
        return [new_con]

    opt = GurobiPersistent_WithImplicitLazyConstraints()
    opt.set_instance(
        model,
        constraint_generator=constraint_generator,
    )

    opt.options["Presolve"] = 0
    opt.options["Heuristics"] = 0

    stream = io.StringIO()
    with capture_output(stream):
        results = opt.solve(tee=True)

    log = stream.getvalue()

    assert results.solver.status == "ok"
    assert results.solver.termination_condition == "optimal"

    assert (
        opt.n_calls > 1
    ), "Expected the constraint generator to be called multiple times."

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
    print("Implicit lazy constraint test passed!")

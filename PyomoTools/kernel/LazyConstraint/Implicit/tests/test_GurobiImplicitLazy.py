from ..GurobiImplicitLazy import GurobiPersistent_WithImplicitLazyConstraints

from pyomo.common.tee import capture_output
import io
import pyomo.kernel as pmo
import numpy as np
import re


def test_InfiniteOpt():
    model = pmo.block()
    mag = 9
    generated_constraint_coeffs = []

    model.x = pmo.variable(lb=-mag, ub=mag, domain=pmo.Integers)
    model.y = pmo.variable(lb=-mag, ub=mag, domain=pmo.Integers)
    model.obj = pmo.objective(expr=model.x + model.y, sense=pmo.maximize)

    # X and Y are bound to within the unit circle. But we will enforce via lazy constraints.
    def constraint_generator(model, tol=1e-2):
        # RULE: (x,y) will almost certainly violate the unit circle. If so, we should be able to draw a line between the incumbent solution and the origin. This line will intersect the unit circle. We will find this point, generate a tangent constraint and add it as a lazy constraint.
        # This will go on until the incumbent solution is within the unit circle or is within the given tolerance (euclidean distance) of the unit circle.

        x = model.x.value
        y = model.y.value

        mag_i = np.sqrt(x**2 + y**2)
        if mag_i <= mag + tol:
            return []

        theta = np.arctan2(y, x)
        a = float(np.cos(theta))
        b = float(np.sin(theta))
        generated_constraint_coeffs.append((a, b, mag))
        new_con = model.x * a + model.y * b <= mag
        return [new_con]

    opt = GurobiPersistent_WithImplicitLazyConstraints()
    opt.set_instance(model, constraint_generator)

    # Deactivate prosolve and heuristics so Guorbi doesn't pre-solve around the lazy constraint
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
    ), "Expected the constraint generator to be called at least twice."

    lazy_constraint_pattern = re.compile(r"Lazy constraints:\s+(\d+)")
    match = lazy_constraint_pattern.search(log)
    assert (
        match is not None
    ), "Could not find 'Lazy constraints' in the log which indicates that the lazy constraints were not processed."
    n_lazy_constraints = int(match.group(1))
    assert (
        n_lazy_constraints > 0
    ), "Expected at least one lazy constraint to be added, but found none."

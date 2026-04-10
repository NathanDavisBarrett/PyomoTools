from ..GurobiExplicitLazy import GurobiPersistent_WithExplicitLazyConstraints
from ...LazyConstraint import lazy_constraint

from pyomo.common.tee import capture_output
import io
import pyomo.kernel as pmo
import numpy as np
import re


def test_InfiniteOpt():
    model = pmo.block()
    mag = 9

    model.x = pmo.variable(lb=-mag, ub=mag, domain=pmo.Integers)
    model.y = pmo.variable(lb=-mag, ub=mag, domain=pmo.Integers)
    model.obj = pmo.objective(expr=model.x + model.y, sense=pmo.maximize)

    # X and Y are bound to within the unit circle. But we will enforce via a bunch of (lazy) linear constraints
    n_points = 10
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    model.c = pmo.constraint_list(
        [
            lazy_constraint(
                expr=model.x * np.cos(angle) + model.y * np.sin(angle) <= mag
            )
            for angle in angles
        ]
    )

    opt = GurobiPersistent_WithExplicitLazyConstraints()
    opt.set_instance(model)

    # Deactivate prosolve and heuristics so Guorbi doesn't pre-solve around the lazy constraint
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

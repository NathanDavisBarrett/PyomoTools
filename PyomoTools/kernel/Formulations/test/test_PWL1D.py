from ..PWL1D import PWL1D, PWL1DParameters

import pyomo.kernel as pmo
from ....base.Solvers import DefaultSolver


def test_general():
    model = pmo.block()
    model.x = pmo.variable()
    model.y = pmo.variable()

    params = PWL1DParameters(points=[(0, 0), (5, 5), (10, 0)])

    model.pwl = PWL1D(
        params,
        model.x,
        model.y,
    )

    model.c1 = pmo.constraint(model.y == model.x - 5)

    model.obj = pmo.objective(model.y, sense=pmo.maximize)

    solver = DefaultSolver("MILP")
    result = solver.solve(model)

    assert result.solver.termination_condition == pmo.TerminationCondition.optimal
    assert abs(pmo.value(model.x) - 7.5) <= 1e-5
    assert abs(pmo.value(model.y) - 2.5) <= 1e-5


def test_concave():
    model = pmo.block()
    model.x = pmo.variable()
    model.y = pmo.variable()

    params = PWL1DParameters(points=[(0, 0), (5, 5), (10, 0)], includeLB_y=False)

    model.pwl = PWL1D(
        params,
        model.x,
        model.y,
    )

    model.c1 = pmo.constraint(model.y == model.x - 5)
    model.c2 = pmo.constraint(model.y == -model.x + 5)

    model.obj = pmo.objective(model.y, sense=pmo.maximize)

    solver = DefaultSolver("MILP")
    result = solver.solve(model)

    assert result.solver.termination_condition == pmo.TerminationCondition.optimal
    assert abs(pmo.value(model.x) - 5.0) <= 1e-5
    assert abs(pmo.value(model.y) - 0.0) <= 1e-5


def test_convex():
    model = pmo.block()
    model.x = pmo.variable()
    model.y = pmo.variable()

    params = PWL1DParameters(points=[(0, 5), (5, 0), (10, 5)], includeUB_y=False)

    model.pwl = PWL1D(
        params,
        model.x,
        model.y,
    )

    model.c1 = pmo.constraint(model.x == 3.5)
    model.c2 = pmo.constraint(model.y == 4.1)

    model.obj = pmo.objective(model.y, sense=pmo.maximize)

    solver = DefaultSolver("MILP")
    result = solver.solve(model)

    assert result.solver.termination_condition == pmo.TerminationCondition.optimal

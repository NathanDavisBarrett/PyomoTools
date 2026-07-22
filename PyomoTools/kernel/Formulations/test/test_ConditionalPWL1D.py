import numpy as np

import pyomo.kernel as pmo

from ..PWL1D import ConditionalPWL1D, XBoundOption
from ....base.Formulations.PWL1D import PWL1DParameters
from ....base.Solvers import DefaultSolver

TOL = 1e-5


def _solve(model):
    solver = DefaultSolver("MILP")
    result = solver.solve(model)
    assert result.solver.termination_condition == pmo.TerminationCondition.optimal
    return result


def test_linear_active_matches_line():
    model = pmo.block()
    model.x = pmo.variable(domain=pmo.Reals, lb=0, ub=4)
    model.y = pmo.variable(domain=pmo.Reals)
    model.condition = pmo.variable(domain=pmo.Binary)

    params = PWL1DParameters.from_linear(2, 1, lb_x=0, ub_x=4)

    model.pwl = ConditionalPWL1D(params, model.x, model.y, model.condition)

    model.condition.fix(1)
    model.x.fix(3)
    model.obj = pmo.objective(model.y, sense=pmo.minimize)

    _solve(model)

    assert np.isclose(pmo.value(model.y), 7.0, atol=TOL)


def test_linear_inactive_forces_zero_output():
    model = pmo.block()
    model.x = pmo.variable(domain=pmo.Reals, lb=0, ub=4)
    model.y = pmo.variable(domain=pmo.Reals)
    model.condition = pmo.variable(domain=pmo.Binary)

    params = PWL1DParameters.from_linear(2, 1, lb_x=0, ub_x=4)

    model.pwl = ConditionalPWL1D(params, model.x, model.y, model.condition)

    model.condition.fix(0)
    model.x.fix(3)
    model.obj = pmo.objective(model.y, sense=pmo.maximize)

    _solve(model)

    assert np.isclose(pmo.value(model.y), 0.0, atol=TOL)


def test_convex_active_recovers_piecewise_value():
    model = pmo.block()
    model.x = pmo.variable(domain=pmo.Reals, lb=0, ub=4)
    model.y = pmo.variable(domain=pmo.Reals)
    model.condition = pmo.variable(domain=pmo.Binary)

    params = PWL1DParameters(points=[(0, 0), (2, 1), (4, 4)], includeUB_y=False)

    model.pwl = ConditionalPWL1D(params, model.x, model.y, model.condition)

    model.condition.fix(1)
    model.x.fix(3)
    model.obj = pmo.objective(model.y, sense=pmo.minimize)

    _solve(model)

    assert np.isclose(pmo.value(model.y), 2.5, atol=TOL)


def test_convex_inactive_forces_zero_output():
    model = pmo.block()
    model.x = pmo.variable(domain=pmo.Reals, lb=0, ub=4)
    model.y = pmo.variable(domain=pmo.Reals)
    model.condition = pmo.variable(domain=pmo.Binary)

    params = PWL1DParameters(points=[(0, 0), (2, 1), (4, 4)], includeUB_y=False)

    model.pwl = ConditionalPWL1D(params, model.x, model.y, model.condition)

    model.condition.fix(0)
    model.x.fix(3)
    model.obj = pmo.objective(model.y, sense=pmo.minimize)

    _solve(model)

    assert np.isclose(pmo.value(model.y), 0.0, atol=TOL)


def test_concave_active_recovers_piecewise_value():
    model = pmo.block()
    model.x = pmo.variable(domain=pmo.Reals, lb=0, ub=4)
    model.y = pmo.variable(domain=pmo.Reals)
    model.condition = pmo.variable(domain=pmo.Binary)

    params = PWL1DParameters(points=[(0, 0), (2, 4), (4, 5)], includeLB_y=False)

    model.pwl = ConditionalPWL1D(params, model.x, model.y, model.condition)

    model.condition.fix(1)
    model.x.fix(3)
    model.obj = pmo.objective(model.y, sense=pmo.maximize)

    _solve(model)

    assert np.isclose(pmo.value(model.y), 4.5, atol=TOL)


def test_concave_inactive_forces_zero_output():
    model = pmo.block()
    model.x = pmo.variable(domain=pmo.Reals, lb=0, ub=4)
    model.y = pmo.variable(domain=pmo.Reals)
    model.condition = pmo.variable(domain=pmo.Binary)

    params = PWL1DParameters(points=[(0, 0), (2, 4), (4, 5)], includeLB_y=False)

    model.pwl = ConditionalPWL1D(params, model.x, model.y, model.condition)

    model.condition.fix(0)
    model.x.fix(3)
    model.obj = pmo.objective(model.y, sense=pmo.maximize)

    _solve(model)

    assert np.isclose(pmo.value(model.y), 0.0, atol=TOL)


def test_general_active_recovers_exact_interpolation():
    model = pmo.block()
    model.x = pmo.variable(domain=pmo.Reals, lb=0, ub=4)
    model.y = pmo.variable(domain=pmo.Reals)
    model.condition = pmo.variable(domain=pmo.Binary)

    params = PWL1DParameters(points=[(0, 0), (2, 3), (4, 1)])

    model.pwl = ConditionalPWL1D(params, model.x, model.y, model.condition)

    model.condition.fix(1)
    model.x.fix(3)
    model.obj = pmo.objective(model.y, sense=pmo.maximize)

    _solve(model)

    assert np.isclose(pmo.value(model.y), 2.0, atol=TOL)


def test_general_inactive_forces_zero_output():
    model = pmo.block()
    model.x = pmo.variable(domain=pmo.Reals, lb=0, ub=4)
    model.y = pmo.variable(domain=pmo.Reals)
    model.condition = pmo.variable(domain=pmo.Binary)

    params = PWL1DParameters(points=[(0, 0), (2, 3), (4, 1)])

    model.pwl = ConditionalPWL1D(params, model.x, model.y, model.condition)

    model.condition.fix(0)
    model.x.fix(3)
    model.obj = pmo.objective(model.y, sense=pmo.maximize)

    _solve(model)

    assert np.isclose(pmo.value(model.y), 0.0, atol=TOL)


def test_point_bound_option_limits_inactive_x_range():
    model = pmo.block()
    model.x = pmo.variable(domain=pmo.Reals, lb=0, ub=10)
    model.y = pmo.variable(domain=pmo.Reals)
    model.condition = pmo.variable(domain=pmo.Binary)

    params = PWL1DParameters(points=[(2, 0), (3, 1), (4, 0)])

    model.pwl = ConditionalPWL1D(
        params,
        model.x,
        model.y,
        model.condition,
        xBoundOption=XBoundOption.POINT_BOUND,
    )

    model.condition.fix(0)
    model.obj = pmo.objective(model.x, sense=pmo.maximize)

    _solve(model)

    assert np.isclose(pmo.value(model.x), 4.0, atol=TOL)
    assert np.isclose(pmo.value(model.y), 0.0, atol=TOL)

from ..PWL1D import PWL1D
from ....base.Formulations.PWL1D import PWL1DParameters

import pyomo.environ as pyo
from ....base.Solvers import DefaultSolver


def test_general():
    """Test general PWL function using SOS2 formulation."""
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.y = pyo.Var()

    params = PWL1DParameters(points=[(0, 0), (5, 5), (10, 0)])

    weights, weightSumConstraint, sos2Constraint, xValueConstraint, yValueConstraint = (
        PWL1D(
            model,
            params,
            model.x,
            model.y,
        )
    )

    model.c1 = pyo.Constraint(expr=model.y == model.x - 5)

    model.obj = pyo.Objective(expr=model.y, sense=pyo.maximize)

    solver = DefaultSolver("MILP")
    result = solver.solve(model, tee=False)

    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    assert abs(pyo.value(model.x) - 7.5) <= 1e-5
    assert abs(pyo.value(model.y) - 2.5) <= 1e-5


def test_concave():
    """Test concave PWL function using inequality constraints."""
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.y = pyo.Var()

    params = PWL1DParameters(points=[(0, 0), (5, 5), (10, 0)], includeLB_y=False)

    (concaveInequalities,) = PWL1D(
        model,
        params,
        model.x,
        model.y,
    )

    model.c1 = pyo.Constraint(expr=model.y == model.x - 5)
    model.c2 = pyo.Constraint(expr=model.y == -model.x + 5)

    model.obj = pyo.Objective(expr=model.y, sense=pyo.maximize)

    solver = DefaultSolver("MILP")
    result = solver.solve(model, tee=False)

    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    assert abs(pyo.value(model.x) - 5.0) <= 1e-5
    assert abs(pyo.value(model.y) - 0.0) <= 1e-5


def test_convex():
    """Test convex PWL function using inequality constraints."""
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.y = pyo.Var()

    params = PWL1DParameters(points=[(0, 5), (5, 0), (10, 5)], includeUB_y=False)

    (convexInequalities,) = PWL1D(
        model,
        params,
        model.x,
        model.y,
    )

    model.c1 = pyo.Constraint(expr=model.x == 3.5)
    model.c2 = pyo.Constraint(expr=model.y == 4.1)

    model.obj = pyo.Objective(expr=model.y, sense=pyo.maximize)

    solver = DefaultSolver("MILP")
    result = solver.solve(model, tee=False)

    assert result.solver.termination_condition == pyo.TerminationCondition.optimal


def test_linear():
    """Test linear PWL function (two points only)."""
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.y = pyo.Var()

    params = PWL1DParameters(points=[(0, 0), (5, 10)])

    (linearEquality,) = PWL1D(
        model,
        params,
        model.x,
        model.y,
    )

    model.c1 = pyo.Constraint(expr=model.x == 2.5)

    model.obj = pyo.Objective(expr=model.y, sense=pyo.maximize)

    solver = DefaultSolver("LP")
    result = solver.solve(model, tee=False)

    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    assert abs(pyo.value(model.x) - 2.5) <= 1e-5
    assert abs(pyo.value(model.y) - 5.0) <= 1e-5  # y = 2*x, so y = 2*2.5 = 5


def test_construction():
    """Test that constraints are properly created and named."""
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.y = pyo.Var()

    # Test general PWL
    params_general = PWL1DParameters(points=[(0, 0), (5, 5), (10, 0)])
    weights, weightSumConstraint, sos2Constraint, xValueConstraint, yValueConstraint = (
        PWL1D(
            model,
            params_general,
            model.x,
            model.y,
        )
    )

    # Check that constraints were created
    assert hasattr(model, "x_y_PWL1D_weights")
    assert hasattr(model, "x_y_PWL1D_weightSum")
    assert hasattr(model, "x_y_PWL1D_xValue")
    assert hasattr(model, "x_y_PWL1D_yValue")

    # Test linear PWL
    model2 = pyo.ConcreteModel()
    model2.a = pyo.Var()
    model2.b = pyo.Var()

    params_linear = PWL1DParameters(points=[(0, 0), (1, 1)])
    (linearEquality,) = PWL1D(
        model2,
        params_linear,
        model2.a,
        model2.b,
    )

    # Check that constraint was created
    assert hasattr(model2, "a_b_PWL1D_linearEquality")


def test_indexed():
    """Test PWL function with indexed variables."""
    model = pyo.ConcreteModel()
    model.I = pyo.Set(initialize=["A", "B"])
    model.x = pyo.Var(model.I)
    model.y = pyo.Var(model.I)

    # Create a dictionary of parameters, one for each index
    params = {
        "A": PWL1DParameters(points=[(0, 0), (5, 5), (10, 0)]),
        "B": PWL1DParameters(points=[(0, 0), (5, 5), (10, 0)]),
    }

    PWL1D(
        model,
        params,
        model.x,
        model.y,
        itrSet=model.I,
    )

    # Add constraints for y values
    model.c1 = pyo.Constraint(expr=model.y["A"] == model.x["A"] - 5)
    model.c2 = pyo.Constraint(expr=model.y["B"] == model.x["B"] - 2.5)

    model.obj = pyo.Objective(expr=model.y["A"] + model.y["B"], sense=pyo.maximize)

    solver = DefaultSolver("MILP")
    result = solver.solve(model, tee=False)

    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    assert abs(pyo.value(model.y["A"]) - 2.5) <= 1e-5
    assert abs(pyo.value(model.y["B"]) - 3.75) <= 1e-5


def test_custom_relationship_name():
    """Test that custom relationship base names work correctly."""
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.y = pyo.Var()

    params = PWL1DParameters(points=[(0, 0), (1, 1)])

    (linearEquality,) = PWL1D(
        model,
        params,
        model.x,
        model.y,
        relationshipBaseName="custom_pwl",
    )

    # Check that constraint was created with custom name
    assert hasattr(model, "custom_pwl_linearEquality")
    assert not hasattr(model, "x_y_PWL1D_linearEquality")


def test_parameter_validation():
    """Test that parameter validation works correctly."""
    model = pyo.ConcreteModel()
    model.I = pyo.Set(initialize=["A", "B"])
    model.x = pyo.Var(model.I)
    model.y = pyo.Var(model.I)

    # Test 1: Non-indexed case with dictionary should raise error
    params_dict = {
        "A": PWL1DParameters(points=[(0, 0), (1, 1)]),
        "B": PWL1DParameters(points=[(0, 0), (1, 1)]),
    }

    try:
        PWL1D(model, params_dict, model.x, model.y, itrSet=None)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "should be a single PWL1DParameters object" in str(e)

    # Test 2: Indexed case with single params should raise error
    model2 = pyo.ConcreteModel()
    model2.I = pyo.Set(initialize=["A", "B"])
    model2.x = pyo.Var(model2.I)
    model2.y = pyo.Var(model2.I)

    params_single = PWL1DParameters(points=[(0, 0), (1, 1)])

    try:
        PWL1D(model2, params_single, model2.x, model2.y, itrSet=model2.I)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "should be a dictionary mapping each index to PWL1DParameters" in str(e)

    # Test 3: Missing index in dictionary should raise error
    model3 = pyo.ConcreteModel()
    model3.I = pyo.Set(initialize=["A", "B", "C"])
    model3.x = pyo.Var(model3.I)
    model3.y = pyo.Var(model3.I)

    params_incomplete = {
        "A": PWL1DParameters(points=[(0, 0), (1, 1)]),
        "B": PWL1DParameters(points=[(0, 0), (1, 1)]),
        # Missing "C"
    }

    try:
        PWL1D(model3, params_incomplete, model3.x, model3.y, itrSet=model3.I)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Missing PWL1DParameters for index C" in str(e)

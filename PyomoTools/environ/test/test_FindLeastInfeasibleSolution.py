import pyomo.environ as pyo
import numpy as np

from ..FindLeastInfeasibleSolution import (
    FindLeastInfeasibleSolution,
    LeastInfeasibleDefinition,
    MapSpecificConstraint,
)
from ...base.Solvers import DefaultSolver


def test_SimpleProblem1():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(0, None))

    model.c1 = pyo.Constraint(expr=model.x <= -1)

    FindLeastInfeasibleSolution(model, DefaultSolver("LP"), tee=True)

    xVal = pyo.value(model.x)
    assert xVal >= -1.000001
    assert xVal <= 0.000001


def test_SimpleProblem_KnownSolution():
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.y = pyo.Var()

    model.c1 = pyo.Constraint(expr=model.y >= 2)
    model.c2 = pyo.Constraint(expr=model.y >= -model.x + 4)
    model.c3 = pyo.Constraint(expr=model.y <= -model.x + 2)
    model.c4 = pyo.Constraint(expr=model.y <= 1)

    FindLeastInfeasibleSolution(model, DefaultSolver("QP"), tee=True)

    # Any point on the line y = x in 1 <= x <= 2 is a valid solution.

    xVal = pyo.value(model.x)
    yVal = pyo.value(model.y)

    assert np.allclose(
        [
            xVal,
        ],
        [
            yVal,
        ],
    )
    assert xVal >= -0.9999999
    assert xVal <= 2.0000001


def test_L2():
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.y = pyo.Var()

    model.c1 = pyo.Constraint(expr=model.y >= 2)
    model.c2 = pyo.Constraint(expr=model.y >= -model.x + 4)
    model.c3 = pyo.Constraint(expr=model.y <= -model.x + 2)
    model.c4 = pyo.Constraint(expr=model.y <= 1)

    FindLeastInfeasibleSolution(
        model,
        DefaultSolver("QP"),
        leastInfeasibleDefinition=LeastInfeasibleDefinition.L2_Norm,
    )

    xVal = pyo.value(model.x)
    yVal = pyo.value(model.y)

    assert np.allclose([xVal, yVal], [1.5, 1.5])


def test_FeasibleProblem():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(-1, 0))

    FindLeastInfeasibleSolution(model, DefaultSolver("LP"), tee=True)

    xVal = pyo.value(model.x)
    assert xVal >= -1.000001
    assert xVal <= 0.000001


def test_Indexed():
    model = pyo.ConcreteModel()
    model.Set1 = pyo.Set(initialize=["A", "B", "C"])
    model.x = pyo.Var(model.Set1, bounds=(-1, 1))
    model.y = pyo.Var(bounds=(1, 2))

    model.c1 = pyo.Constraint(model.Set1, rule=lambda _, i: model.x[i] == model.y)
    model.c2 = pyo.Constraint(expr=model.y == sum(model.x[i] for i in model.Set1))

    FindLeastInfeasibleSolution(model, DefaultSolver("LP"), tee=True)

    results = [pyo.value(model.y), *[pyo.value(model.x[i]) for i in model.Set1]]
    assert np.allclose(results, np.zeros(len(results)))


def test_RelaxOnlySpecificConstraint():
    """Test relaxing only one specific constraint while keeping others enforced."""
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.y = pyo.Var()

    # Create multiple constraints, only one is infeasible
    model.c1 = pyo.Constraint(expr=model.x >= 0)  # Feasible
    model.c2 = pyo.Constraint(expr=model.y >= 0)  # Feasible
    model.c3 = pyo.Constraint(expr=model.x + model.y <= -1)  # Infeasible

    # Relax only c3
    FindLeastInfeasibleSolution(
        model, DefaultSolver("LP"), relax_only_these_constraints=[model.c3], tee=True
    )

    xVal = pyo.value(model.x)
    yVal = pyo.value(model.y)

    # x and y should still respect their bounds (>= 0) since those weren't relaxed
    assert xVal >= -0.000001
    assert yVal >= -0.000001


def test_RelaxOnlyVariableBounds():
    """Test relaxing only specific variable bounds."""
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(0, None))
    # Create a constraint that forces x to be negative
    model.c1 = pyo.Constraint(expr=model.x <= -1)

    # Relax only x's bounds, not the constraint
    FindLeastInfeasibleSolution(
        model, DefaultSolver("LP"), relax_only_these_constraints=[model.x], tee=True
    )

    xVal = pyo.value(model.x)

    # x should satisfy the constraint (x <= -1)
    assert xVal <= -1 + 0.000001


def test_RelaxIndexedConstraint():
    """Test relaxing an indexed constraint."""
    model = pyo.ConcreteModel()
    model.Set1 = pyo.Set(initialize=[0, 1, 2])
    model.x = pyo.Var(model.Set1, bounds=(0, None))

    model.c = pyo.Constraint(model.Set1, rule=lambda _, i: model.x[i] <= -1)

    # Relax the indexed constraint
    FindLeastInfeasibleSolution(
        model, DefaultSolver("LP"), relax_only_these_constraints=[model.c], tee=True
    )

    for i in model.Set1:
        xVal = pyo.value(model.x[i])
        # Should find a solution respecting variable bounds (>= 0)
        assert xVal >= -0.000001
        assert xVal <= 0.000001


def test_RelaxIndexedVariable():
    """Test relaxing bounds on an indexed variable."""
    model = pyo.ConcreteModel()
    model.Set1 = pyo.Set(initialize=[0, 1, 2])
    model.x = pyo.Var(model.Set1, bounds=(0, None))

    model.c = pyo.Constraint(model.Set1, rule=lambda _, i: model.x[i] <= -1)

    # Relax only the variable bounds
    FindLeastInfeasibleSolution(
        model, DefaultSolver("LP"), relax_only_these_constraints=[model.x], tee=True
    )

    for i in model.Set1:
        xVal = pyo.value(model.x[i])
        # Should satisfy constraints (x[i] <= -1)
        assert xVal <= -1 + 0.000001


def test_RelaxSpecificBlock():
    """Test relaxing all constraints within a specific sub-block."""
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(0, None))
    model.sub = pyo.Block()
    model.sub.y = pyo.Var(bounds=(0, None))

    # Constraint in main block
    model.c1 = pyo.Constraint(expr=model.x >= 5)

    # Infeasible constraint in sub-block
    model.sub.c1 = pyo.Constraint(expr=model.sub.y <= -1)

    # Relax only the sub-block
    FindLeastInfeasibleSolution(
        model, DefaultSolver("LP"), relax_only_these_constraints=[model.sub], tee=True
    )

    xVal = pyo.value(model.x)
    yVal = pyo.value(model.sub.y)

    # x should satisfy its non-relaxed constraint
    assert xVal >= 5 - 0.000001

    # y's constraint and bounds were relaxed, should be at 0
    assert yVal >= -1.000001
    assert yVal <= 0.000001


def test_RelaxMultipleSpecificConstraints():
    """Test relaxing multiple specific constraints."""
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.y = pyo.Var()
    model.z = pyo.Var(bounds=(0, None))

    model.c1 = pyo.Constraint(expr=model.x >= 10)
    model.c2 = pyo.Constraint(expr=model.y <= -10)
    model.c3 = pyo.Constraint(expr=model.x + model.y == 0)

    # Relax c1 and c2, but not c3
    FindLeastInfeasibleSolution(
        model,
        DefaultSolver("LP"),
        relax_only_these_constraints=[model.c1, model.c2],
        tee=True,
    )

    xVal = pyo.value(model.x)
    yVal = pyo.value(model.y)

    # c3 should still be satisfied
    assert np.allclose(xVal + yVal, 0, atol=1e-5)


def test_RelaxMixedConstraintsAndBounds():
    """Test relaxing a mix of constraints and variable bounds."""
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(0, 1))
    model.y = pyo.Var(bounds=(0, None))

    model.c1 = pyo.Constraint(expr=model.x + model.y >= 5)
    model.c2 = pyo.Constraint(expr=model.x <= -1)

    # Relax c2 and x's bounds
    FindLeastInfeasibleSolution(
        model,
        DefaultSolver("LP"),
        relax_only_these_constraints=[model.c2, model.x],
        tee=True,
    )

    xVal = pyo.value(model.x)
    yVal = pyo.value(model.y)

    # c1 should still be satisfied (non-relaxed)
    assert xVal + yVal >= 5 - 0.000001

    # y's bounds were not relaxed
    assert yVal >= -0.000001


def test_PartialRelaxation_StillInfeasible():
    """Test that if we only relax some constraints and the problem is still infeasible, an error is raised."""
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(0, None))

    model.c1 = pyo.Constraint(expr=model.x >= 10)
    model.c2 = pyo.Constraint(expr=model.x <= -10)

    # Only relax c1, leaving c2 which creates an infeasible problem
    try:
        FindLeastInfeasibleSolution(
            model,
            DefaultSolver("LP"),
            relax_only_these_constraints=[model.c1],
            tee=True,
        )
        # If we get here, the test should fail
        assert False, "Expected an exception for remaining infeasibility"
    except Exception as e:
        # Should get an error message mentioning partial relaxation
        assert "subset of constraints were relaxed" in str(e)


def test_MapSpecificConstraint_SimpleConstraint():
    """Test that MapSpecificConstraint can map a simple constraint."""
    from copy import deepcopy

    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.c1 = pyo.Constraint(expr=model.x >= 0)

    # Clone the model (simulates what FindLeastInfeasibleSolution does)
    cloned_model = deepcopy(model)

    # Use MapSpecificConstraint to find c1 in the cloned model
    mapped_constraint = MapSpecificConstraint(model, cloned_model, model.c1.name)

    # Verify we got the right constraint
    assert isinstance(mapped_constraint, pyo.Constraint)
    assert mapped_constraint.local_name == "c1"
    assert mapped_constraint is cloned_model.c1


def test_MapSpecificConstraint_IndexedConstraint():
    """Test that MapSpecificConstraint can map indexed constraints."""
    from copy import deepcopy

    model = pyo.ConcreteModel()
    model.Set1 = pyo.Set(initialize=[0, 1, 2])
    model.x = pyo.Var(model.Set1)
    model.c = pyo.Constraint(model.Set1, rule=lambda _, i: model.x[i] >= 0)

    cloned_model = deepcopy(model)

    # Map each indexed constraint
    for i in model.Set1:
        mapped_constraint = MapSpecificConstraint(model, cloned_model, model.c[i].name)
        assert isinstance(mapped_constraint, pyo.Constraint)
        assert mapped_constraint is cloned_model.c[i]


def test_MapSpecificConstraint_NestedBlock():
    """Test that MapSpecificConstraint can navigate nested blocks."""
    from copy import deepcopy

    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.sub = pyo.Block()
    model.sub.y = pyo.Var()
    model.sub.c1 = pyo.Constraint(expr=model.sub.y >= 0)

    cloned_model = deepcopy(model)

    # Map the constraint in the sub-block
    mapped_constraint = MapSpecificConstraint(model, cloned_model, model.sub.c1.name)

    assert isinstance(mapped_constraint, pyo.Constraint)
    assert mapped_constraint.local_name == "c1"
    assert mapped_constraint is cloned_model.sub.c1


def test_MapSpecificConstraint_DeeplyNested():
    """Test that MapSpecificConstraint can handle deeply nested structures."""
    from copy import deepcopy

    model = pyo.ConcreteModel()
    model.level1 = pyo.Block()
    model.level1.level2 = pyo.Block()
    model.level1.level2.x = pyo.Var()
    model.level1.level2.c = pyo.Constraint(expr=model.level1.level2.x >= 0)

    cloned_model = deepcopy(model)

    # Map the deeply nested constraint
    mapped_constraint = MapSpecificConstraint(
        model, cloned_model, model.level1.level2.c.name
    )

    assert isinstance(mapped_constraint, pyo.Constraint)
    assert mapped_constraint.local_name == "c"
    assert mapped_constraint is cloned_model.level1.level2.c

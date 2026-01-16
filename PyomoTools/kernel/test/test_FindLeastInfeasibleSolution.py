import pyomo.kernel as pmo
import numpy as np

from ..FindLeastInfeasibleSolution import (
    FindLeastInfeasibleSolution,
    LeastInfeasibleDefinition,
    MapSpecificConstraint,
)
from ...base.Solvers import DefaultSolver


def test_SimpleProblem1():
    model = pmo.block()
    model.x = pmo.variable(lb=0)

    model.c1 = pmo.constraint(model.x <= -1)

    FindLeastInfeasibleSolution(model, DefaultSolver("LP"), tee=True)

    xVal = pmo.value(model.x)
    assert xVal >= -1.000001
    assert xVal <= 0.000001


def test_SimpleProblem_KnownSolution():
    model = pmo.block()
    model.x = pmo.variable()
    model.y = pmo.variable()

    model.c1 = pmo.constraint(expr=model.y >= 2)
    model.c2 = pmo.constraint(expr=model.y >= -model.x + 4)
    model.c3 = pmo.constraint(expr=model.y <= -model.x + 2)
    model.c4 = pmo.constraint(expr=model.y <= 1)

    FindLeastInfeasibleSolution(
        model,
        DefaultSolver("LP"),
        tee=True,
        leastInfeasibleDefinition=LeastInfeasibleDefinition.L2_Norm,
    )

    # Any point on the line y = x in 1 <= x <= 2 is a valid solution.

    xVal = pmo.value(model.x)
    yVal = pmo.value(model.y)

    assert np.allclose([xVal, yVal], [1.5, 1.5])


def test_L2():
    model = pmo.block()
    model.x = pmo.variable()
    model.y = pmo.variable()

    model.c1 = pmo.constraint(expr=model.y >= 2)
    model.c2 = pmo.constraint(expr=model.y >= -model.x + 4)
    model.c3 = pmo.constraint(expr=model.y <= -model.x + 2)
    model.c4 = pmo.constraint(expr=model.y <= 1)

    FindLeastInfeasibleSolution(
        model,
        DefaultSolver("QP"),
        leastInfeasibleDefinition=LeastInfeasibleDefinition.L2_Norm,
    )

    xVal = pmo.value(model.x)
    yVal = pmo.value(model.y)

    assert np.allclose([xVal, yVal], [1.5, 1.5])


def test_FeasibleProblem():
    model = pmo.block()
    model.x = pmo.variable(lb=-1, ub=0)

    FindLeastInfeasibleSolution(model, DefaultSolver("LP"), tee=True)

    xVal = pmo.value(model.x)
    assert xVal >= -1.000001
    assert xVal <= 0.000001


def test_Indexed():
    model = pmo.block()
    model.x = pmo.variable_list([pmo.variable(lb=0) for i in range(3)])

    model.c1 = pmo.constraint_list([pmo.constraint(model.x[i] <= -1) for i in range(3)])

    FindLeastInfeasibleSolution(model, DefaultSolver("LP"), tee=True)

    for i in range(3):
        xVal = pmo.value(model.x[i])
        assert xVal >= -1.000001
        assert xVal <= 0.000001


def test_Multilevel():
    model = pmo.block()
    model.x = pmo.variable(lb=0)
    model.sub = pmo.block()
    model.sub.x = pmo.variable(lb=0)

    model.c1 = pmo.constraint(model.x <= -1)
    model.sub.c1 = pmo.constraint(model.sub.x <= -1)

    FindLeastInfeasibleSolution(model, DefaultSolver("LP"), tee=True)
    xVal = pmo.value(model.x)
    assert xVal >= -1.000001
    assert xVal <= 0.000001

    xVal = pmo.value(model.sub.x)
    assert xVal >= -1.000001
    assert xVal <= 0.000001


def test_RelaxOnlySpecificConstraint():
    """Test relaxing only one specific constraint while keeping others enforced."""
    model = pmo.block()
    model.x = pmo.variable()
    model.y = pmo.variable()

    # Create multiple constraints, only one is infeasible
    model.c1 = pmo.constraint(expr=model.x >= 0)  # Feasible
    model.c2 = pmo.constraint(expr=model.y >= 0)  # Feasible
    model.c3 = pmo.constraint(expr=model.x + model.y <= -1)  # Infeasible

    # Relax only c3
    FindLeastInfeasibleSolution(
        model, DefaultSolver("LP"), relax_only_these_constraints=[model.c3], tee=True
    )

    xVal = pmo.value(model.x)
    yVal = pmo.value(model.y)

    # x and y should still respect their bounds (>= 0) since those weren't relaxed
    assert xVal >= -0.000001
    assert yVal >= -0.000001


def test_RelaxOnlyVariableBounds():
    """Test relaxing only specific variable bounds."""
    model = pmo.block()
    model.x = pmo.variable(lb=0)
    # Create a constraint that forces x to be negative
    model.c1 = pmo.constraint(expr=model.x <= -1)

    # Relax only x's bounds, not the constraint
    FindLeastInfeasibleSolution(
        model, DefaultSolver("LP"), relax_only_these_constraints=[model.x], tee=True
    )

    xVal = pmo.value(model.x)

    # x should satisfy the constraint (x <= -1)
    assert xVal <= -1 + 0.000001


def test_RelaxConstraintList():
    """Test relaxing an entire constraint_list."""
    model = pmo.block()
    model.x = pmo.variable_list([pmo.variable(lb=0) for i in range(3)])

    model.c = pmo.constraint_list(
        [
            pmo.constraint(model.x[0] <= -1),
            pmo.constraint(model.x[1] <= -1),
            pmo.constraint(model.x[2] <= -1),
        ]
    )

    # Relax the entire constraint list
    FindLeastInfeasibleSolution(
        model, DefaultSolver("LP"), relax_only_these_constraints=[model.c], tee=True
    )

    for i in range(3):
        xVal = pmo.value(model.x[i])
        # Should find a solution respecting variable bounds (>= 0)
        assert xVal >= -0.000001
        assert xVal <= 0.000001


def test_RelaxVariableList():
    """Test relaxing bounds on an entire variable_list."""
    model = pmo.block()
    model.x = pmo.variable_list([pmo.variable(lb=0) for i in range(3)])

    model.c = pmo.constraint_list([pmo.constraint(model.x[i] <= -1) for i in range(3)])

    # Relax only the variable bounds
    FindLeastInfeasibleSolution(
        model, DefaultSolver("LP"), relax_only_these_constraints=[model.x], tee=True
    )

    for i in range(3):
        xVal = pmo.value(model.x[i])
        # Should satisfy constraints (x[i] <= -1)
        assert xVal <= -1 + 0.000001


def test_RelaxSpecificBlock():
    """Test relaxing all constraints within a specific sub-block."""
    model = pmo.block()
    model.x = pmo.variable(lb=0)
    model.sub = pmo.block()
    model.sub.y = pmo.variable(lb=0)

    # Constraint in main block
    model.c1 = pmo.constraint(expr=model.x >= 5)

    # Infeasible constraint in sub-block
    model.sub.c1 = pmo.constraint(expr=model.sub.y <= -1)

    # Relax only the sub-block
    FindLeastInfeasibleSolution(
        model, DefaultSolver("LP"), relax_only_these_constraints=[model.sub], tee=True
    )

    xVal = pmo.value(model.x)
    yVal = pmo.value(model.sub.y)

    # x should satisfy its non-relaxed constraint
    assert xVal >= 5 - 0.000001

    # y's constraint and bounds were relaxed, should be at 0
    assert yVal >= -1.000001
    assert yVal <= 0.000001


def test_RelaxMultipleSpecificConstraints():
    """Test relaxing multiple specific constraints."""
    model = pmo.block()
    model.x = pmo.variable()
    model.y = pmo.variable()
    model.z = pmo.variable(lb=0)

    model.c1 = pmo.constraint(expr=model.x >= 10)
    model.c2 = pmo.constraint(expr=model.y <= -10)
    model.c3 = pmo.constraint(expr=model.x + model.y == 0)

    # Relax c1 and c2, but not c3
    FindLeastInfeasibleSolution(
        model,
        DefaultSolver("LP"),
        relax_only_these_constraints=[model.c1, model.c2],
        tee=True,
    )

    xVal = pmo.value(model.x)
    yVal = pmo.value(model.y)

    # c3 should still be satisfied
    assert np.allclose(xVal + yVal, 0, atol=1e-5)


def test_RelaxMixedConstraintsAndBounds():
    """Test relaxing a mix of constraints and variable bounds."""
    model = pmo.block()
    model.x = pmo.variable(lb=0, ub=1)
    model.y = pmo.variable(lb=0)

    model.c1 = pmo.constraint(expr=model.x + model.y >= 5)
    model.c2 = pmo.constraint(expr=model.x <= -1)

    # Relax c2 and x's bounds
    FindLeastInfeasibleSolution(
        model,
        DefaultSolver("LP"),
        relax_only_these_constraints=[model.c2, model.x],
        tee=True,
    )

    xVal = pmo.value(model.x)
    yVal = pmo.value(model.y)

    # c1 should still be satisfied (non-relaxed)
    assert xVal + yVal >= 5 - 0.000001

    # y's bounds were not relaxed
    assert yVal >= -0.000001


def test_RelaxBlockList():
    """Test relaxing constraints in a block_list."""
    model = pmo.block()
    model.x = pmo.variable(lb=0)

    model.sub = pmo.block_list()
    for i in range(2):
        b = pmo.block()
        b.y = pmo.variable(lb=0)
        b.c = pmo.constraint(expr=b.y <= -1)
        model.sub.append(b)

    model.c_main = pmo.constraint(expr=model.x >= 10)

    # Relax only the block_list
    FindLeastInfeasibleSolution(
        model, DefaultSolver("LP"), relax_only_these_constraints=[model.sub], tee=True
    )

    xVal = pmo.value(model.x)
    # Main constraint should still be satisfied
    assert xVal >= 10 - 0.000001

    # Sub-block constraints were relaxed
    for i in range(2):
        yVal = pmo.value(model.sub[i].y)
        assert yVal >= -1.000001
        assert yVal <= 0.000001


def test_PartialRelaxation_StillInfeasible():
    """Test that if we only relax some constraints and the problem is still infeasible, an error is raised."""
    model = pmo.block()
    model.x = pmo.variable(domain=pmo.NonNegativeReals)

    model.c1 = pmo.constraint(expr=model.x >= 10)
    model.c2 = pmo.constraint(expr=model.x <= -10)

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
    model = pmo.block()
    model.x = pmo.variable()
    model.c1 = pmo.constraint(expr=model.x >= 0)

    # Clone the model (simulates what FindLeastInfeasibleSolution does)
    cloned_model = model.clone()

    # Use MapSpecificConstraint to find c1 in the cloned model
    mapped_constraint = MapSpecificConstraint(model, cloned_model, model.c1.name)

    # Verify we got the right constraint
    assert isinstance(mapped_constraint, pmo.constraint)
    assert mapped_constraint.local_name == "c1"
    assert mapped_constraint is cloned_model.c1


def test_MapSpecificConstraint_IndexedConstraint():
    """Test that MapSpecificConstraint can map indexed constraints."""
    model = pmo.block()
    model.x = pmo.variable_list([pmo.variable() for i in range(3)])
    model.c = pmo.constraint_list(
        [pmo.constraint(expr=model.x[i] >= 0) for i in range(3)]
    )

    cloned_model = model.clone()

    # Map each indexed constraint
    for i in range(3):
        mapped_constraint = MapSpecificConstraint(model, cloned_model, model.c[i].name)
        assert isinstance(mapped_constraint, pmo.constraint)
        assert mapped_constraint is cloned_model.c[i]


def test_MapSpecificConstraint_NestedBlock():
    """Test that MapSpecificConstraint can navigate nested blocks."""
    model = pmo.block()
    model.x = pmo.variable()
    model.sub = pmo.block()
    model.sub.y = pmo.variable()
    model.sub.c1 = pmo.constraint(expr=model.sub.y >= 0)

    cloned_model = model.clone()

    # Map the constraint in the sub-block
    mapped_constraint = MapSpecificConstraint(model, cloned_model, model.sub.c1.name)

    assert isinstance(mapped_constraint, pmo.constraint)
    assert mapped_constraint.local_name == "c1"
    assert mapped_constraint is cloned_model.sub.c1


def test_MapSpecificConstraint_IndexedBlockWithConstraint():
    """Test that MapSpecificConstraint can navigate indexed blocks."""
    model = pmo.block()
    model.sub = pmo.block_list()
    for i in range(2):
        b = pmo.block()
        b.x = pmo.variable()
        b.c = pmo.constraint(expr=b.x >= 0)
        model.sub.append(b)

    cloned_model = model.clone()

    # Map constraints in indexed blocks
    for i in range(2):
        mapped_constraint = MapSpecificConstraint(
            model, cloned_model, model.sub[i].c.name
        )
        assert isinstance(mapped_constraint, pmo.constraint)
        assert mapped_constraint is cloned_model.sub[i].c


def test_MapSpecificConstraint_DeeplyNested():
    """Test that MapSpecificConstraint can handle deeply nested structures."""
    model = pmo.block()
    model.level1 = pmo.block()
    model.level1.level2 = pmo.block()
    model.level1.level2.x = pmo.variable()
    model.level1.level2.c = pmo.constraint(expr=model.level1.level2.x >= 0)

    cloned_model = model.clone()

    # Map the deeply nested constraint
    mapped_constraint = MapSpecificConstraint(
        model, cloned_model, model.level1.level2.c.name
    )

    assert isinstance(mapped_constraint, pmo.constraint)
    assert mapped_constraint.local_name == "c"
    assert mapped_constraint is cloned_model.level1.level2.c


def test_MapSpecificConstraint_ComplexIndexing():
    """Test MapSpecificConstraint with complex indexed structures."""
    model = pmo.block()
    model.blocks = pmo.block_list()
    for i in range(2):
        b = pmo.block()
        b.x = pmo.variable()
        b.constraints = pmo.constraint_list(
            [pmo.constraint(expr=b.x >= i + j) for j in range(3)]
        )
        model.blocks.append(b)

    cloned_model = model.clone()

    # Map constraints in indexed blocks with indexed constraint lists
    for i in range(2):
        for j in range(3):
            mapped_constraint = MapSpecificConstraint(
                model, cloned_model, model.blocks[i].constraints[j].name
            )
            assert isinstance(mapped_constraint, pmo.constraint)
            assert mapped_constraint is cloned_model.blocks[i].constraints[j]


def test_RetryOriginalObjective():
    """Test that retry_original_objective optimizes the original objective while maintaining minimal infeasibility.

    This test uses conflicting constraints where multiple solutions achieve the same
    minimum L1-norm, but the original objective prefers one solution over others.
    """
    model = pmo.block()
    model.x = pmo.variable(lb=0, ub=10)
    model.y = pmo.variable(lb=0, ub=10)

    # Create conflicting constraints:
    # c1 wants x >= 8, c3 wants x <= 3 → any x ∈ [3,8] gives total slack of 5
    # c2 wants y >= 8, c4 wants y <= 3 → any y ∈ [3,8] gives total slack of 5
    # Total minimum slack = 10, achieved by any (x,y) in [3,8] × [3,8]
    model.c1 = pmo.constraint(expr=model.x >= 8)
    model.c2 = pmo.constraint(expr=model.y >= 8)
    model.c3 = pmo.constraint(expr=model.x <= 3)
    model.c4 = pmo.constraint(expr=model.y <= 3)

    # Original objective: minimize x + 2*y
    # Among all solutions with minimal infeasibility, this prefers x=3, y=3
    model.obj = pmo.objective(expr=model.x + 2 * model.y, sense=pmo.minimize)

    # First, test without retry_original_objective
    model_copy1 = model.clone()
    FindLeastInfeasibleSolution(
        model_copy1, DefaultSolver("LP"), retry_original_objective=False, tee=True
    )

    # Now test with retry_original_objective
    model_copy2 = model.clone()
    FindLeastInfeasibleSolution(
        model_copy2, DefaultSolver("LP"), retry_original_objective=True, tee=True
    )

    x1, y1 = pmo.value(model_copy1.x), pmo.value(model_copy1.y)
    x2, y2 = pmo.value(model_copy2.x), pmo.value(model_copy2.y)

    # Calculate total slack for both solutions
    slack1 = abs(x1 - 8) + abs(x1 - 3) + abs(y1 - 8) + abs(y1 - 3)
    slack2 = abs(x2 - 8) + abs(x2 - 3) + abs(y2 - 8) + abs(y2 - 3)

    # Both should have the same level of infeasibility (total slack = 10)
    assert np.allclose(slack1, 10, atol=1e-4)
    assert np.allclose(slack2, 10, atol=1e-4)

    # The objective value with retry should be better (lower) or equal
    obj1 = x1 + 2 * y1
    obj2 = x2 + 2 * y2
    assert obj2 <= obj1 + 1e-4

    # With retry_original_objective, we should get x≈3, y≈3 to minimize x+2y
    assert np.allclose(x2, 3, atol=1e-4)
    assert np.allclose(y2, 3, atol=1e-4)

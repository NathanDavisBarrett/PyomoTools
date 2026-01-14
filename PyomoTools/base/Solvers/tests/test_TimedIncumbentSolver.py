from ..TimedIncumbentSolver import TimedIncumbentSolver

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import numpy as np


def test_Simple():

    # Pyomo model
    model = pyo.ConcreteModel()
    model.x = pyo.Var(within=pyo.Binary)
    model.y = pyo.Var(within=pyo.NonNegativeReals)
    model.obj = pyo.Objective(expr=2 * model.x + model.y, sense=pyo.maximize)
    model.c1 = pyo.Constraint(expr=model.x + model.y <= 1.5)

    solver = TimedIncumbentSolver(5)
    results = solver.solve(model)

    assert results.solver.termination_condition == TerminationCondition.optimal

    xVal = pyo.value(model.x)
    yVal = pyo.value(model.y)

    assert np.allclose(np.array([xVal, yVal]), np.array([1.0, 0.5]))


def test_FindFirstIncumbent():
    """
    Test with a challenging Set Cover problem that takes time to solve.
    We create a set cover instance where we need to select subsets to cover all elements,
    with overlapping coverage and varying costs.
    """
    np.random.seed(42)

    # Problem size
    n_elements = 300  # Elements to cover
    n_subsets = 500  # Available subsets

    model = pyo.ConcreteModel()

    model.Elements = pyo.RangeSet(1, n_elements)
    model.Subsets = pyo.RangeSet(1, n_subsets)

    # Binary variable: whether to select each subset
    model.x = pyo.Var(model.Subsets, within=pyo.Binary)

    # Generate coverage matrix: each subset covers random elements
    # Make coverage sparse but ensure feasibility
    coverage = {}
    for j in model.Subsets:
        # Each subset covers between 5-15 random elements
        n_covered = np.random.randint(5, 16)
        covered_elements = np.random.choice(
            list(model.Elements), n_covered, replace=False
        )
        for i in covered_elements:
            coverage[i, j] = 1

    # Ensure every element can be covered by at least 2 subsets for feasibility
    for i in model.Elements:
        covering_subsets = [j for j in model.Subsets if (i, j) in coverage]
        while len(covering_subsets) < 2:
            j = np.random.choice(list(model.Subsets))
            if (i, j) not in coverage:
                coverage[i, j] = 1
                covering_subsets.append(j)

    # Costs: most subsets have cost 1, some have higher costs to make problem harder
    costs = {j: 1 + np.random.exponential(0.5) for j in model.Subsets}

    # Objective: minimize total cost
    model.obj = pyo.Objective(
        expr=sum(costs[j] * model.x[j] for j in model.Subsets), sense=pyo.minimize
    )

    # Constraints: each element must be covered at least once
    model.coverage_constraints = pyo.ConstraintList()
    for i in model.Elements:
        covering_subsets = [j for j in model.Subsets if (i, j) in coverage]
        model.coverage_constraints.add(
            expr=sum(model.x[j] for j in covering_subsets) >= 1
        )

    # Additional constraints to make problem harder: limit total selections
    # This creates a tighter feasible region
    model.budget = pyo.Constraint(
        expr=sum(model.x[j] for j in model.Subsets) <= n_elements * 0.4
    )

    solver = TimedIncumbentSolver(timeLimit=0)
    results = solver.solve(model, tee=True)

    # Check that a solution was found
    assert results.solver.termination_condition in [
        TerminationCondition.maxTimeLimit,
        TerminationCondition.optimal,
        TerminationCondition.feasible,
    ]
    assert results.solver.status == results.solver.status.__class__.ok

    # Verify at least one subset was selected
    selected = sum(1 for j in model.Subsets if pyo.value(model.x[j]) > 0.5)
    assert selected > 0
    print(f"Selected {selected} subsets")

import pyomo.environ as pyo

from ..ConstrainContiguousBinarySeries import ConstrainContiguousBinarySeries
from PyomoTools.base.Solvers import DefaultSolver, WrappedSolver


def GetBinarySeries(n: int, mx: bool = True):
    m = pyo.ConcreteModel()
    m.X = pyo.Var(range(n), domain=pyo.Binary)

    if mx:
        m.obj = pyo.Objective(expr=sum(m.X[i] for i in range(n)), sense=pyo.maximize)
    else:
        m.obj = pyo.Objective(expr=sum(m.X[i] for i in range(n)), sense=pyo.minimize)

    return m


def executeSetIndicesTest(n: int, start: int, end: int):
    m = GetBinarySeries(n)
    startIndex, endIndex, constraints = ConstrainContiguousBinarySeries(
        m, [m.X[i] for i in range(n)]
    )

    # Set indices
    startIndex.fix(start)
    endIndex.fix(end)

    solver = WrappedSolver(DefaultSolver("MILP"))
    results = solver.solve(m, tee=False)
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal

    from PyomoTools.environ import InfeasibilityReport

    rep = InfeasibilityReport(m, onlyInfeasibilities=False)
    rep.WriteFile("infeasibilityReport.txt")

    # Check if the binary variables are set correctly
    for i in range(n):
        if start <= i <= end:
            assert (
                pyo.value(m.X[i]) == 1
            ), f"X[{i}] should be 1 but is {pyo.value(m.X[i])}"
        else:
            assert (
                pyo.value(m.X[i]) == 0
            ), f"X[{i}] should be 0 but is {pyo.value(m.X[i])}"


def test_MiddleIndices():
    executeSetIndicesTest(10, 3, 6)


def test_ZeroStart():
    executeSetIndicesTest(10, 0, 5)


def test_TerminalEnd():
    executeSetIndicesTest(10, 5, 9)


def test_Alternating():
    n = 10
    m = GetBinarySeries(n)
    startIndex, endIndex, constraints = ConstrainContiguousBinarySeries(
        m, [m.X[i] for i in range(n)]
    )

    m.X[3].fix(1)
    m.X[4].fix(0)
    m.X[5].fix(1)

    solver = DefaultSolver("MILP")
    results = solver.solve(m, tee=False)
    assert len(results.solution) == 0, "Expected zero solutions"


def test_FillBetween():
    n = 10
    m = GetBinarySeries(n)
    startIndex, endIndex, constraints = ConstrainContiguousBinarySeries(
        m, [m.X[i] for i in range(n)]
    )

    m.X[3].fix(1)
    m.X[8].fix(1)

    solver = DefaultSolver("MILP")
    results = solver.solve(m, tee=False)
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal

    # Check if the binary variables are set correctly
    for i in range(3, 9):
        assert pyo.value(m.X[i]) == 1, f"X[{i}] should be 1 but is {pyo.value(m.X[i])}"


def test_DetermineStartStop():
    n = 10
    m = GetBinarySeries(n)
    startIndex, endIndex, constraints = ConstrainContiguousBinarySeries(
        m, [m.X[i] for i in range(n)]
    )

    m.X[0].fix(0)
    m.X[4].fix(1)
    m.X[9].fix(0)

    solver = DefaultSolver("MILP")
    results = solver.solve(m, tee=False)
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal

    start_val = pyo.value(startIndex)
    end_val = pyo.value(endIndex)

    assert start_val >= 0 and start_val <= 4
    assert end_val >= 4 and end_val <= 9

import pyomo.kernel as pmo

from ..ConstrainContiguousBinarySeries import ConstrainContiguousBinarySeries
from PyomoTools.base.Solvers import DefaultSolver, WrappedSolver


def GetBinarySeries(n: int, mx: bool = True):
    m = pmo.block()
    m.X = pmo.variable_list([pmo.variable(domain=pmo.Binary) for _ in range(n)])

    if mx:
        m.obj = pmo.objective(expr=sum(m.X), sense=pmo.maximize)
    else:
        m.obj = pmo.objective(expr=sum(m.X), sense=pmo.minimize)

    return m


def executeSetIndicesTest(n: int, start: int, end: int):
    m = GetBinarySeries(n)
    m.c = ConstrainContiguousBinarySeries(m.X)

    # Set indices
    m.c.StartIndex.fix(start)
    m.c.EndIndex.fix(end)

    solver = WrappedSolver(DefaultSolver("MILP"))
    results = solver.solve(m, tee=False)
    assert results.solver.termination_condition == pmo.TerminationCondition.optimal

    from PyomoTools.kernel import InfeasibilityReport

    rep = InfeasibilityReport(m, onlyInfeasibilities=False)
    rep.WriteFile("infeasibilityReport.txt")

    # Check if the binary variables are set correctly
    for i in range(n):
        if start <= i <= end:
            assert m.X[i].value == 1, f"X[{i}] should be 1 but is {m.X[i].value}"
        else:
            assert m.X[i].value == 0, f"X[{i}] should be 0 but is {m.X[i].value}"


def test_MiddleIndices():
    executeSetIndicesTest(10, 3, 6)


def test_ZeroStart():
    executeSetIndicesTest(10, 0, 5)


def test_TerminalEnd():
    executeSetIndicesTest(10, 5, 9)


def test_Alternating():
    n = 10
    m = GetBinarySeries(n)
    m.c = ConstrainContiguousBinarySeries(m.X)

    m.X[3].fix(1)
    m.X[4].fix(0)
    m.X[5].fix(1)

    solver = DefaultSolver("MILP")
    results = solver.solve(m, tee=False)
    assert len(results.solution) == 0, "Expected zero solutions"


def test_FillBetween():
    n = 10
    m = GetBinarySeries(n)
    m.c = ConstrainContiguousBinarySeries(m.X)

    m.X[3].fix(1)
    m.X[8].fix(1)

    solver = DefaultSolver("MILP")
    results = solver.solve(m, tee=False)
    assert results.solver.termination_condition == pmo.TerminationCondition.optimal

    # Check if the binary variables are set correctly
    for i in range(3, 9):
        assert m.X[i].value == 1, f"X[{i}] should be 1 but is {m.X[i].value}"


def test_DetermineStartStop():
    n = 10
    m = GetBinarySeries(n)
    m.c = ConstrainContiguousBinarySeries(m.X)

    m.X[0].fix(0)
    m.X[4].fix(1)
    m.X[9].fix(0)

    solver = DefaultSolver("MILP")
    results = solver.solve(m, tee=False)
    assert results.solver.termination_condition == pmo.TerminationCondition.optimal

    start = pmo.value(m.c.StartIndex)
    end = pmo.value(m.c.EndIndex)

    assert start >= 0 and start <= 4
    assert end >= 4 and end <= 9

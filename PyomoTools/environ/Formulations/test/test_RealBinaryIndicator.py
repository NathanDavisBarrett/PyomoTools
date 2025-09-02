from ..RealBinaryIndicator import RealBinaryIndicator, Relation
from ....base.Solvers import DefaultSolver

import pyomo.environ as pyo


def test_Construction():
    m = pyo.ConcreteModel()
    m.X = pyo.Var(domain=pyo.Binary)
    m.Y = pyo.Var(domain=pyo.Binary)
    m.A = pyo.Var()
    alphaPrime = 5.0
    alphaMin = 1.0
    alphaMax = 10.0
    constraints = RealBinaryIndicator(
        m,
        m.X,
        m.Y,
        m.A,
        alphaPrime=alphaPrime,
        alphaMin=alphaMin,
        alphaMax=alphaMax,
        aRelation=Relation.GEQ,
    )

    # Check that constraints were created
    assert len(constraints) > 0


def executePointTest(
    Xval,
    YVal,
    Aval,
    alphaPrime,
    alphaMin,
    alphaMax,
    aRelation: Relation,
    enforceBinary: bool = False,
) -> bool:
    """
    A function to test whether or not a given point is feasible in the generated model.
    """
    m = pyo.ConcreteModel()

    if enforceBinary:
        m.X = pyo.Var(domain=pyo.Binary)
        m.Y = pyo.Var(domain=pyo.Binary)
    else:
        m.X = pyo.Var(domain=pyo.Reals, bounds=(0, 1))
        m.Y = pyo.Var(domain=pyo.Reals, bounds=(0, 1))

    m.A = pyo.Var()

    RealBinaryIndicator(
        m,
        m.X,
        m.Y,
        m.A,
        alphaPrime=alphaPrime,
        alphaMin=alphaMin,
        alphaMax=alphaMax,
        aRelation=aRelation,
    )

    # Fix the variables to the test values
    m.X.fix(Xval)
    m.Y.fix(YVal)
    m.A.fix(Aval)

    m.junkObj = pyo.Objective(
        expr=m.X, sense=pyo.minimize
    )  # Dummy objective to allow solving

    if enforceBinary:
        solver = DefaultSolver("MILP")
    else:
        solver = DefaultSolver("LP")

    results = solver.solve(m, tee=False)

    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        return True
    else:
        return False


def test_GEQ_ExtremePoints():
    """
    Test the extreme points of the GEQ relation
    """
    alphaPrime = 5.0
    alphaMin = 1.0
    alphaMax = 10.0

    extremePoints = [
        (0, 0, alphaMin),
        (0, 1, alphaMin),
        (1, 1, alphaPrime),
        (0, 0, alphaMax),
        # (0,1,alphaPrime), #This point is not feasible due to epsilon
        (1, 1, alphaMax),
    ]
    for Xval, YVal, Aval in extremePoints:
        assert executePointTest(
            Xval,
            YVal,
            Aval,
            alphaPrime,
            alphaMin,
            alphaMax,
            Relation.GEQ,
            enforceBinary=True,
        )


def test_LEQ_ExtremePoints():
    """
    Test the extreme points of the LEQ relation
    """
    alphaPrime = 5.0
    alphaMin = 1.0
    alphaMax = 10.0

    extremePoints = [
        (0, 0, alphaMin),
        (0, 1, alphaMax),
        (1, 1, alphaMin),
        (0, 0, alphaMax),
        (1, 1, alphaPrime),
    ]
    for Xval, YVal, Aval in extremePoints:
        assert executePointTest(
            Xval,
            YVal,
            Aval,
            alphaPrime,
            alphaMin,
            alphaMax,
            Relation.LEQ,
            enforceBinary=True,
        )


def test_GEQ_PointsOfInterest():
    """
    Test points of interest that should be feasible or infeasible for GEQ relation
    """
    alphaPrime = 5.0
    alphaMin = 1.0
    alphaMax = 10.0

    # Points that should be feasible
    feasiblePoints = [
        (0, 0, 3.0),  # X=0, Y=0, any A
        (0, 1, 2.0),  # X=0, Y=1, A < alphaPrime
        (1, 1, 7.0),  # X=1, Y=1, A >= alphaPrime
    ]

    for Xval, YVal, Aval in feasiblePoints:
        assert executePointTest(
            Xval,
            YVal,
            Aval,
            alphaPrime,
            alphaMin,
            alphaMax,
            Relation.GEQ,
            enforceBinary=True,
        ), f"Point ({Xval}, {YVal}, {Aval}) should be feasible"

    # Points that should be infeasible
    infeasiblePoints = [
        (1, 0, 3.0),  # X=1, Y=0 (violates the relationship)
        (
            1,
            1,
            3.0,
        ),  # X=1, Y=1, A < alphaPrime (violates A >= alphaPrime when X=1, Y=1)
    ]

    for Xval, YVal, Aval in infeasiblePoints:
        assert not executePointTest(
            Xval,
            YVal,
            Aval,
            alphaPrime,
            alphaMin,
            alphaMax,
            Relation.GEQ,
            enforceBinary=True,
        ), f"Point ({Xval}, {YVal}, {Aval}) should be infeasible"


def test_LEQ_PointsOfInterest():
    """
    Test points of interest that should be feasible or infeasible for LEQ relation
    """
    alphaPrime = 5.0
    alphaMin = 1.0
    alphaMax = 10.0

    # Points that should be feasible
    feasiblePoints = [
        (0, 0, 7.0),  # X=0, Y=0, any A
        (0, 1, 8.0),  # X=0, Y=1, A > alphaPrime
        (1, 1, 3.0),  # X=1, Y=1, A <= alphaPrime
    ]

    for Xval, YVal, Aval in feasiblePoints:
        assert executePointTest(
            Xval,
            YVal,
            Aval,
            alphaPrime,
            alphaMin,
            alphaMax,
            Relation.LEQ,
            enforceBinary=True,
        ), f"Point ({Xval}, {YVal}, {Aval}) should be feasible"

    # Points that should be infeasible
    infeasiblePoints = [
        (1, 0, 3.0),  # X=1, Y=0 (violates the relationship)
        (
            1,
            1,
            8.0,
        ),  # X=1, Y=1, A > alphaPrime (violates A <= alphaPrime when X=1, Y=1)
    ]

    for Xval, YVal, Aval in infeasiblePoints:
        assert not executePointTest(
            Xval,
            YVal,
            Aval,
            alphaPrime,
            alphaMin,
            alphaMax,
            Relation.LEQ,
            enforceBinary=True,
        ), f"Point ({Xval}, {YVal}, {Aval}) should be infeasible"


def test_EdgeCases_GEQ():
    """
    Test edge cases where alphaPrime is at the boundaries
    """
    alphaMin = 1.0
    alphaMax = 10.0

    # Case 1: alphaPrime > alphaMax (X should always be 0)
    alphaPrime = 15.0
    m = pyo.ConcreteModel()
    m.X = pyo.Var(domain=pyo.Binary)
    m.Y = pyo.Var(domain=pyo.Binary)
    m.A = pyo.Var(bounds=(alphaMin, alphaMax))

    RealBinaryIndicator(
        m,
        m.X,
        m.Y,
        m.A,
        alphaPrime=alphaPrime,
        alphaMin=alphaMin,
        alphaMax=alphaMax,
        aRelation=Relation.GEQ,
    )

    m.Y.fix(1)
    m.A.fix(5.0)
    m.obj = pyo.Objective(expr=m.X, sense=pyo.maximize)

    solver = DefaultSolver("MILP")
    results = solver.solve(m, tee=False)
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal
    assert pyo.value(m.X) == 0  # X should be 0

    # Case 2: alphaPrime < alphaMin (X should equal Y)
    alphaPrime = -1.0
    m = pyo.ConcreteModel()
    m.X = pyo.Var(domain=pyo.Binary)
    m.Y = pyo.Var(domain=pyo.Binary)
    m.A = pyo.Var(bounds=(alphaMin, alphaMax))

    RealBinaryIndicator(
        m,
        m.X,
        m.Y,
        m.A,
        alphaPrime=alphaPrime,
        alphaMin=alphaMin,
        alphaMax=alphaMax,
        aRelation=Relation.GEQ,
    )

    m.Y.fix(1)
    m.A.fix(5.0)
    m.obj = pyo.Objective(expr=m.X, sense=pyo.minimize)

    solver = DefaultSolver("MILP")
    results = solver.solve(m, tee=False)
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal
    assert pyo.value(m.X) == 1  # X should equal Y

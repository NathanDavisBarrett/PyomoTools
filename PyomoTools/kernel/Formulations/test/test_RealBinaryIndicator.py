from ..RealBinaryIndicator import RealBinaryIndicator, Relation
from ....base.Solvers import DefaultSolver

import pyomo.kernel as pmo


def test_Construction():
    m = pmo.block()
    X = pmo.variable(domain=pmo.Binary)
    Y = pmo.variable(domain=pmo.Binary)
    A = pmo.variable()
    alphaPrime = 5.0
    alphaMin = 1.0
    alphaMax = 10.0
    m.rbi = RealBinaryIndicator(
        X,
        Y,
        A,
        alphaPrime=alphaPrime,
        alphaMin=alphaMin,
        alphaMax=alphaMax,
        aRelation=Relation.GEQ,
    )

    import matplotlib

    matplotlib.use("TkAgg")
    m.rbi.Plot()


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
    if enforceBinary:
        kwargs = dict(domain=pmo.Binary)
    else:
        kwargs = dict(domain=pmo.Reals, lb=0, ub=1)

    m = pmo.block()
    m.X = pmo.variable(**kwargs)
    m.Y = pmo.variable(**kwargs)
    m.A = pmo.variable()
    m.rbi = RealBinaryIndicator(
        m.X,
        m.Y,
        m.A,
        alphaPrime=alphaPrime,
        alphaMin=alphaMin,
        alphaMax=alphaMax,
        aRelation=aRelation,
    )

    m.FixConstraints = pmo.constraint_list(
        [
            pmo.constraint(expr=m.X == Xval),
            pmo.constraint(expr=m.Y == YVal),
            pmo.constraint(expr=m.A == Aval),
        ]
    )

    m.junkObj = pmo.objective(
        expr=m.X, sense=pmo.minimize
    )  # Dummy objective to allow solving

    if enforceBinary:
        solver = DefaultSolver("MILP")
    else:
        solver = DefaultSolver("LP")

    results = solver.solve(m, tee=False)

    if results.solver.termination_condition == pmo.TerminationCondition.optimal:
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
    Test the extreme points of the GEQ relation
    """
    alphaPrime = 5.0
    alphaMin = 1.0
    alphaMax = 10.0

    extremePoints = [
        (0, 0, alphaMin),
        # (0,1,alphaPrime),  # This point is not feasible due to epsilon
        (1, 1, alphaMin),
        (0, 0, alphaMin),
        (1, 1, alphaMin),
        (0, 0, alphaMax),
        (0, 1, alphaMax),
        (1, 1, alphaPrime),
        (0, 0, alphaMax),
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


def test_GEQ_FeasibleInteriorPoints():
    alphaPrime = 5.0
    alphaMin = 1.0
    alphaMax = 10.0

    a1 = (alphaMax + alphaPrime) / 2.0
    a2 = (alphaMin + alphaPrime) / 2.0

    points = [
        (0.5, 1, alphaPrime),
        (0, 0.5, alphaPrime),
        (0.5, 0.5, alphaPrime),
        (0, 0, a1),
        (0, 0, alphaPrime),
        (0, 0, a2),
        (1, 1, a1),
        (0, 1, a2),
    ]
    for Xval, YVal, Aval in points:
        assert executePointTest(
            Xval,
            YVal,
            Aval,
            alphaPrime,
            alphaMin,
            alphaMax,
            Relation.GEQ,
            enforceBinary=False,
        )


def test_LEQ_FeasibleInteriorPoints():
    alphaPrime = 5.0
    alphaMin = 1.0
    alphaMax = 10.0

    a1 = (alphaMax + alphaPrime) / 2.0
    a2 = (alphaMin + alphaPrime) / 2.0

    points = [
        (0.5, 1, alphaPrime),
        (0, 0.5, alphaPrime),
        (0.5, 0.5, alphaPrime),
        (0, 0, a1),
        (0, 0, alphaPrime),
        (0, 0, a2),
        (0, 1, a1),
        (1, 1, a2),
    ]
    for Xval, YVal, Aval in points:
        assert executePointTest(
            Xval,
            YVal,
            Aval,
            alphaPrime,
            alphaMin,
            alphaMax,
            Relation.LEQ,
            enforceBinary=False,
        )


def test_GEQ_InfeasibleInteriorPoints():
    alphaPrime = 5.0
    alphaMin = 1.0
    alphaMax = 10.0

    points = [
        (1, 0, alphaMin),
        (1, 0, alphaPrime),
        (1, 0, alphaMax),
        (1, 1, alphaMin),
        (0, 1, alphaMax),
        (0, 1, alphaPrime),
    ]
    for Xval, YVal, Aval in points:
        assert not executePointTest(
            Xval,
            YVal,
            Aval,
            alphaPrime,
            alphaMin,
            alphaMax,
            Relation.GEQ,
            enforceBinary=False,
        )


def test_LEQ_InfeasibleInteriorPoints():
    alphaPrime = 5.0
    alphaMin = 1.0
    alphaMax = 10.0

    points = [
        (1, 0, alphaMin),
        (1, 0, alphaPrime),
        (1, 0, alphaMax),
        (0, 1, alphaMin),
        (1, 1, alphaMax),
        (0, 1, alphaPrime),
    ]

    for Xval, YVal, Aval in points:
        assert not executePointTest(
            Xval,
            YVal,
            Aval,
            alphaPrime,
            alphaMin,
            alphaMax,
            Relation.LEQ,
            enforceBinary=False,
        )

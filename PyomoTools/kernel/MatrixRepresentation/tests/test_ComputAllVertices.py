from ..MatrixRepresentation import MatrixRepresentation

import pyomo.kernel as pmo


def test_ComputeAllVertices():
    m = pmo.block()
    m.A = pmo.variable(lb=0, ub=10)
    m.B = pmo.variable(lb=0, ub=10)
    m.C = pmo.variable(lb=0, ub=10)

    m.c1 = pmo.constraint(m.A == 2 * m.B)
    m.c2 = pmo.constraint(m.A >= m.C)
    m.c3 = pmo.constraint(m.B + m.C <= 15)

    mr = MatrixRepresentation(m)
    # print(mr)

    import matplotlib

    matplotlib.use("TkAgg")
    mr.Plot()

from ..OrOperator import OrOperator

import pyomo.kernel as pmo


def test_Construction():
    m = pmo.block()
    A = pmo.variable(domain=pmo.Binary)
    B = pmo.variable(domain=pmo.Binary)
    C = pmo.variable(domain=pmo.Binary)

    m.orOp = OrOperator(A, B, C)

    # import matplotlib
    # matplotlib.use('TkAgg')
    # import matplotlib.pyplot as plt
    # m.orOp.Plot()
    # plt.show()

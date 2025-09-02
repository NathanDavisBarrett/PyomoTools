from ..AndOperator import AndOperator

import pyomo.kernel as pmo


def test_Construction():
    m = pmo.block()
    A = pmo.variable(domain=pmo.Binary)
    B = pmo.variable(domain=pmo.Binary)
    C = pmo.variable(domain=pmo.Binary)

    m.andOp = AndOperator(A, B, C)

    # import matplotlib
    # matplotlib.use('TkAgg')
    # import matplotlib.pyplot as plt
    # m.andOp.Plot()
    # plt.show()

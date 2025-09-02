from ..OrOperator import OrOperator

import pyomo.environ as pyo


def test_Construction():
    m = pyo.ConcreteModel()
    m.A = pyo.Var(domain=pyo.Binary)
    m.B = pyo.Var(domain=pyo.Binary)
    m.C = pyo.Var(domain=pyo.Binary)

    constraints = OrOperator(m, m.A, m.B, m.C)

    # Check that constraints were created
    assert len(constraints) == 3
    assert hasattr(m, "A_B_C_OrOperator_constraint1")
    assert hasattr(m, "A_B_C_OrOperator_constraint2")
    assert hasattr(m, "A_B_C_OrOperator_constraint3")


def test_Construction_Indexed():
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=[1, 2, 3])
    m.A = pyo.Var(m.I, domain=pyo.Binary)
    m.B = pyo.Var(m.I, domain=pyo.Binary)
    m.C = pyo.Var(m.I, domain=pyo.Binary)

    constraints = OrOperator(m, m.A, m.B, m.C, itrSet=m.I)

    # Check that constraints were created
    assert len(constraints) == 3
    assert hasattr(m, "A_B_C_OrOperator_constraint1")
    assert hasattr(m, "A_B_C_OrOperator_constraint2")
    assert hasattr(m, "A_B_C_OrOperator_constraint3")

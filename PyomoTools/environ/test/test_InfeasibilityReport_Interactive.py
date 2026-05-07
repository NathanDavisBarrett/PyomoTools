import pyomo.environ as pyo

# Now import the interactive infeasibility report
from ..InfeasibilityReport_Interactive import (
    InfeasibilityReport_Interactive,
)


def create_simple_feasible_model():
    """Create a simple feasible model for testing."""
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.y = pyo.Var()
    model.c = pyo.Constraint(expr=model.x == 2 * model.y)

    model.x.value = 2.0
    model.y.value = 1.0

    return model


def create_simple_infeasible_model():
    """Create a simple infeasible model for testing."""
    model = pyo.ConcreteModel()
    model.idx = pyo.Set(initialize=[0, 1])
    model.x = pyo.Var(model.idx)

    def c_rule(m, i):
        if i == 0:
            return m.x[0] == m.x[1] * 2
        elif i == 1:
            return m.x[0] == 2.0

    model.c = pyo.Constraint(model.idx, rule=c_rule)

    model.y = pyo.Var()
    model.c2 = pyo.Constraint(expr=model.y == 3 * model.x[0])

    model.x[0].value = 2.0
    model.x[1].value = 2.0  # This makes c[0] infeasible
    model.y.value = 0.0  # This makes c2 infeasible

    return model


def test_Interactive():
    model = create_simple_infeasible_model()
    ir = InfeasibilityReport_Interactive(model)
    ir.show()  # This will display the interactive GUI window

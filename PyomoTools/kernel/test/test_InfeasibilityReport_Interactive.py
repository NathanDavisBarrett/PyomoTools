import pyomo.kernel as pmo

# Now import the interactive infeasibility report
from ..InfeasibilityReport_Interactive import (
    InfeasibilityReport_Interactive,
)


def create_simple_feasible_model():
    """Create a simple feasible model for testing."""
    model = pmo.block()
    model.x = pmo.variable()
    model.y = pmo.variable()
    model.c = pmo.constraint(model.x == 2 * model.y)

    model.x.value = 2.0
    model.y.value = 1.0

    return model


def create_simple_infeasible_model():
    """Create a simple infeasible model for testing."""
    model = pmo.block()
    model.x = pmo.variable_list([pmo.variable(), pmo.variable()])

    model.c = pmo.constraint_list(
        [
            pmo.constraint(model.x[0] == model.x[1] * 2),
            pmo.constraint(model.x[0] == 2.0),
        ]
    )

    model.y = pmo.variable()
    model.c2 = pmo.constraint(model.y == 3 * model.x[0])

    model.x[0].value = 2.0
    model.x[1].value = 2.0  # This makes c[0] infeasible
    model.y.value = 0.0  # This makes c2 infeasible

    return model


def create_hierarchical_infeasible_model():
    """Create a hierarchical model with sub-blocks for testing."""
    model = pmo.block()

    # Root level constraints
    model.x = pmo.variable()
    model.y = pmo.variable()
    model.c1 = pmo.constraint(model.x == 5.0)
    model.c2 = pmo.constraint(model.y == 3.0)

    model.x.value = 2.0  # Violates c1
    model.y.value = 3.0  # Satisfies c2

    # Sub-block with its own constraints
    model.sub1 = pmo.block()
    model.sub1.a = pmo.variable()
    model.sub1.b = pmo.variable()
    model.sub1.sub_c1 = pmo.constraint(model.sub1.a <= 10.0)
    model.sub1.sub_c2 = pmo.constraint(model.sub1.b >= 5.0)

    model.sub1.a.value = 15.0  # Violates sub_c1
    model.sub1.b.value = 3.0  # Violates sub_c2

    # Another sub-block with list constraints
    model.sub2 = pmo.block()
    model.sub2.vars = pmo.variable_list([pmo.variable(), pmo.variable()])
    model.sub2.constrs = pmo.constraint_list(
        [
            pmo.constraint(model.sub2.vars[0] == 1.0),
            pmo.constraint(model.sub2.vars[1] == 2.0),
        ]
    )

    model.sub2.vars[0].value = 1.0  # Satisfies constrs[0]
    model.sub2.vars[1].value = 5.0  # Violates constrs[1]

    return model


def test_Interactive():
    model = create_hierarchical_infeasible_model()
    ir = InfeasibilityReport_Interactive(model)
    ir.show()  # This will display the interactive GUI window

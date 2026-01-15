import pyomo.environ as pyo

from ..InfeasibilityReport import InfeasibilityReport


def test_Basic_Feasible():
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.y = pyo.Var()
    model.c = pyo.Constraint(expr=model.x == 2 * model.y)

    model.x.value = 2.0
    model.y.value = 1.0

    report = InfeasibilityReport(model)
    if len(report) != 0:
        raise AssertionError(
            f"The following infeasibilities were detected:\n{str(report)}"
        )


def test_Indexed_Feasible():
    model = pyo.ConcreteModel()
    model.x = pyo.Var([0, 1])

    def cRule(m, i):
        if i == 0:
            return model.x[0] == 2 * model.x[1]
        else:
            return model.x[0] == 2.0

    model.c = pyo.Constraint([0, 1], rule=cRule)

    model.x[0].value = 2.0
    model.x[1].value = 1.0

    report = InfeasibilityReport(model)
    if len(report) != 0:
        raise AssertionError(
            f"The following infeasibilities were detected:\n{str(report)}"
        )


def test_Infeasible():
    model = pyo.ConcreteModel()
    model.x = pyo.Var([0, 1])

    def cRule(m, i):
        if i == 0:
            return model.x[0] == 2 * model.x[1]
        else:
            return model.x[0] == 2.0

    model.c = pyo.Constraint([0, 1], rule=cRule)

    model.x[0].value = 2.0
    model.x[1].value = 1.0

    model.y = pyo.Var()
    model.c1 = pyo.Constraint(expr=model.y == 3 * model.x[0])

    model.x[0].value = 2.0
    model.x[1].value = 2.0
    model.y.value = 0.0

    report = InfeasibilityReport(model)
    assert len(report) == 2


def assertStringEquals(target, reportStr):
    target = target.rstrip()
    reportStr = reportStr.rstrip()

    targLines = target.split("\n")
    reportLines = reportStr.split("\n")
    numLines = max(len(targLines), len(reportLines))
    for i in range(numLines):
        print(f"{i}:")
        if i < len(targLines):
            targLine = targLines[i].rstrip().expandtabs(tabsize=4)
        else:
            raise AssertionError(
                f"Report contains the following line that the target does not have:\n{reportLines[i].rstrip().expandtabs(tabsize=4)}\n\n\nFull Report Output:\n{reportStr}"
            )

        print(f'\ttargLine  : "{targLine}"')

        if i < len(reportLines):
            reportLine = reportLines[i].rstrip().expandtabs(tabsize=4)
        else:
            raise AssertionError(
                f"Target contains the following line that the report does not have:\n{targLine}\n\n\nFull Report Output:\n{reportStr}"
            )
        print(f'\treportLine: "{reportLine}"')

        numChar = max(len(targLine), len(reportLine))
        diffj = None
        for j in range(numChar):
            if j < len(targLine):
                targC = targLine[j]
            else:
                diffj = j
                break

            if j < len(reportLine):
                reportC = reportLine[j]
            else:
                diffj = j
                break

            if targC != reportC:
                diffj = j
                break

        if diffj is not None:
            prior = " " * (j)

            message = f"Report output does not match expected value at the following position:\n{targLine}\n{prior}^\n{prior}|\n{prior}v\n{reportLine}\n\n\nFull Report Output:\n{reportStr}"
            raise AssertionError(message)


def test_ReportFormat():
    model = pyo.ConcreteModel()
    model.x = pyo.Var([0, 1])

    def cRule(m, i):
        if i == 0:
            return model.x[0] == 2 * model.x[1]
        else:
            return model.x[0] == 2.0

    model.c = pyo.Constraint([0, 1], rule=cRule)

    model.y = pyo.Var()
    model.c1 = pyo.Constraint(expr=model.y == 3 * model.x[0])

    model.x[0].value = 2.0
    model.x[1].value = 2.0
    model.y.value = 0.0

    report = InfeasibilityReport(model)
    reportStr = str(report)
    target = """c[0]: x[0]  ==  2*x[1]
      2.0   ==  2*2.0
      │     │   └─┬─┘
      │    ┌┘┌────┘
      2.0 == 4

c1: y    ==  3*x[0]
    0.0  ==  3*2.0
    │    │   └─┬─┘
    │    │ ┌───┘
    0.0 == 6
"""
    assertStringEquals(target, reportStr)


def test_InactiveConstraint_Ignored():
    """Test that inactive constraints are ignored even if violated."""
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.y = pyo.Var()

    # Create a violated constraint
    model.c = pyo.Constraint(expr=model.x == model.y)

    # Set values that violate the constraint
    model.x.value = 1.0
    model.y.value = 5.0

    # Deactivate the constraint
    model.c.deactivate()

    # Report should be empty since the violated constraint is inactive
    report = InfeasibilityReport(model)
    assert (
        len(report) == 0
    ), f"Expected 0 infeasibilities with inactive constraint, but found {len(report)}"


def test_InactiveConstraint_WithActiveConstraint():
    """Test that inactive constraints are ignored while active ones are reported."""
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.y = pyo.Var()

    # Create two violated constraints
    model.c1 = pyo.Constraint(expr=model.x == model.y)
    model.c2 = pyo.Constraint(expr=model.x == 2 * model.y)

    # Set values that violate both constraints
    model.x.value = 1.0
    model.y.value = 5.0

    # Deactivate the first constraint
    model.c1.deactivate()

    # Report should only have one infeasibility (c2)
    report = InfeasibilityReport(model)
    assert len(report) == 1, f"Expected 1 infeasibility, but found {len(report)}"


def test_InactiveIndexedConstraint_Ignored():
    """Test that individual inactive indexed constraints are ignored."""
    model = pyo.ConcreteModel()
    model.x = pyo.Var([0, 1, 2])

    def cRule(m, i):
        return model.x[i] == i

    model.c = pyo.Constraint([0, 1, 2], rule=cRule)

    # Set values that violate all constraints
    model.x[0].value = 10.0
    model.x[1].value = 20.0
    model.x[2].value = 30.0

    # Deactivate constraints at indices 0 and 2
    model.c[0].deactivate()
    model.c[2].deactivate()

    # Report should only have one infeasibility (index 1)
    report = InfeasibilityReport(model)
    assert len(report) == 1, f"Expected 1 infeasibility, but found {len(report)}"

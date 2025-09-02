import pyomo.kernel as pmo

from ..InfeasibilityReport import InfeasibilityReport


def test_Basic_Feasible():
    model = pmo.block()
    model.x = pmo.variable()
    model.y = pmo.variable()
    model.c = pmo.constraint(model.x == 2 * model.y)

    model.x.value = 2.0
    model.y.value = 1.0

    report = InfeasibilityReport(model)
    if len(report) != 0:
        raise AssertionError(
            f"The following infeasibilities were detected:\n{str(report)}"
        )


def test_List_Feasible():
    model = pmo.block()
    model.x = pmo.variable_list([pmo.variable(), pmo.variable()])

    model.c = pmo.constraint_list(
        [
            pmo.constraint(model.x[0] == model.x[1] * 2),
            pmo.constraint(model.x[0] == 2.0),
        ]
    )

    model.x[0].value = 2.0
    model.x[1].value = 1.0

    report = InfeasibilityReport(model)
    if len(report) != 0:
        raise AssertionError(
            f"The following infeasibilities were detected:\n{str(report)}"
        )


def test_Tuple_Feasible():
    model = pmo.block()
    model.x = pmo.variable_tuple((pmo.variable(), pmo.variable()))

    model.c = pmo.constraint_tuple(
        (
            pmo.constraint(model.x[0] == model.x[1] * 2),
            pmo.constraint(model.x[0] == 2.0),
        )
    )

    model.x[0].value = 2.0
    model.x[1].value = 1.0

    report = InfeasibilityReport(model)
    if len(report) != 0:
        raise AssertionError(
            f"The following infeasibilities were detected:\n{str(report)}"
        )


def test_Dict_Feasible():
    model = pmo.block()
    model.x = pmo.variable_dict({"0": pmo.variable(), "1": pmo.variable()})

    model.c = pmo.constraint_dict(
        {
            "0": pmo.constraint(model.x["0"] == model.x["1"] * 2),
            "1": pmo.constraint(model.x["0"] == 2.0),
        }
    )

    model.x["0"].value = 2.0
    model.x["1"].value = 1.0

    report = InfeasibilityReport(model)
    if len(report) != 0:
        raise AssertionError(
            f"The following infeasibilities were detected:\n{str(report)}"
        )


def test_Infeasible():
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
        # print(f"{i}:")
        if i < len(targLines):
            targLine = targLines[i].rstrip().expandtabs(tabsize=4)
        else:
            raise AssertionError(
                f"Report contains the following line that the target does not have:\n{reportLines[i].rstrip().expandtabs(tabsize=4)}\n\n\nFull Report Output:\n{reportStr}"
            )

        # print(f"\ttargLine  : \"{targLine}\"")

        if i < len(reportLines):
            reportLine = reportLines[i].rstrip().expandtabs(tabsize=4)
        else:
            raise AssertionError(
                f"Target contains the following line that the report does not have:\n{targLine}\n\n\nFull Report Output:\n{reportStr}"
            )
        # print(f"\treportLine: \"{reportLine}\"")

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


def fillModelWithInfeasible(model):
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
    model.x[1].value = 2.0
    model.y.value = 0.0


def test_ReportFormat():
    model = pmo.block()
    fillModelWithInfeasible(model)

    report = InfeasibilityReport(model)
    reportStr = str(report)
    target = """ROOT
| c[0]: x[0] - 2*x[1]  ==  0.0
|       2.0  - 2*2.0   ==  0.0
|       2.0 - 2*2.0 == 0.0
|       -2.0 == 0.0
| 
| c2: y   - 3*x[0]  ==  0.0
|     0.0 - 3*2.0   ==  0.0
|     0.0 - 3*2.0 == 0.0
|     -6.0 == 0.0
|
"""
    assertStringEquals(target, reportStr)


def test_ReportFormat_Multilevel():
    model = pmo.block()
    fillModelWithInfeasible(model)

    model.sub = pmo.block()
    fillModelWithInfeasible(model.sub)

    model.sub.sub = pmo.block()
    fillModelWithInfeasible(model.sub.sub)

    model.sub.interConstr = pmo.constraint(model.sub.x[0] == model.sub.sub.x[0] + 2.0)

    report = InfeasibilityReport(model)

    model.subList = pmo.block_list([pmo.block(), pmo.block()])
    fillModelWithInfeasible(model.subList[0])
    fillModelWithInfeasible(model.subList[1])

    report = InfeasibilityReport(model)
    reportStr = str(report)
    target = """ROOT
| c[0]: x[0] - 2*x[1]  ==  0.0
|       2.0  - 2*2.0   ==  0.0
|       2.0 - 2*2.0 == 0.0
|       -2.0 == 0.0
| 
| c2: y   - 3*x[0]  ==  0.0
|     0.0 - 3*2.0   ==  0.0
|     0.0 - 3*2.0 == 0.0
|     -6.0 == 0.0
|
| sub
| | sub.c[0]: sub.x[0] - 2*sub.x[1]  ==  0.0
| |           2.0      - 2*2.0       ==  0.0
| |           2.0 - 2*2.0 == 0.0
| |           -2.0 == 0.0
| | 
| | sub.c2: sub.y - 3*sub.x[0]  ==  0.0
| |         0.0   - 3*2.0       ==  0.0
| |         0.0 - 3*2.0 == 0.0
| |         -6.0 == 0.0
| |
| | sub.interConstr: sub.x[0] - (sub.sub.x[0] + 2.0)  ==  0.0
| |                  2.0      - (2.0          + 2.0)  ==  0.0
| |                  2.0 - (2.0 + 2.0) == 0.0
| |                  -2.0 == 0.0
| |
| | sub.sub
| | | sub.sub.c[0]: sub.sub.x[0] - 2*sub.sub.x[1]  ==  0.0
| | |               2.0          - 2*2.0           ==  0.0
| | |               2.0 - 2*2.0 == 0.0
| | |               -2.0 == 0.0
| | | 
| | | sub.sub.c2: sub.sub.y - 3*sub.sub.x[0]  ==  0.0
| | |             0.0       - 3*2.0           ==  0.0
| | |             0.0 - 3*2.0 == 0.0
| | |             -6.0 == 0.0
| | |
| |
|
| subList[0]
| | subList[0].c[0]: subList[0].x[0] - 2*subList[0].x[1]  ==  0.0
| |                  2.0             - 2*2.0              ==  0.0
| |                  2.0 - 2*2.0 == 0.0
| |                  -2.0 == 0.0
| | 
| | subList[0].c2: subList[0].y - 3*subList[0].x[0]  ==  0.0
| |                0.0          - 3*2.0              ==  0.0
| |                0.0 - 3*2.0 == 0.0
| |                -6.0 == 0.0
| |
|
| subList[1]
| | subList[1].c[0]: subList[1].x[0] - 2*subList[1].x[1]  ==  0.0
| |                  2.0             - 2*2.0              ==  0.0
| |                  2.0 - 2*2.0 == 0.0
| |                  -2.0 == 0.0
| | 
| | subList[1].c2: subList[1].y - 3*subList[1].x[0]  ==  0.0
| |                0.0          - 3*2.0              ==  0.0
| |                0.0 - 3*2.0 == 0.0
| |                -6.0 == 0.0
| |
|
"""
    assertStringEquals(target, reportStr)

import pyomo.environ as pyo
from inspect import isfunction
import matplotlib.pyplot as plt
import numpy as np
from pyomo.core import Piecewise
from pwlf import PiecewiseLinFit


def PWL(
    model: pyo.ConcreteModel,
    func,
    xVar: pyo.Var,
    yVar: pyo.Var,
    xBounds: list,
    numSegments: int,
    relation="==",
    args=(),
    kwargs={},
    verify=True,
    formulation="DCC",
    numFineResolutionSegments=50,
):
    """
    A function that creates a PWL approximation of the function provided.

    Parameters
    ----------
    model: pyo.ConcreteModel
        The model within which you'd like to implement this PWL relationship
    func: python function OR a numpy array
        The nonlinear function you'd like to approximate using PWL. This should be a python function that takes in the value "x" (along with some other optional arguments or key-word arguments) and returns the value "y".

        Alternatively you could pass a numpy array of shape (n,2) where n is a number of points. The first column should be all the x-coordinates of the points of the function. The second column should be all the y-coordinates.
    xVar: pyo.Var
        The pyomo variable that represents "x" in this relationship
    yVar: pyo.Var
        The pyomo variable that represents "y" in this relationship
    xbounds: list
        A list of two floats representing the minimum and maximum possible values of "x"
    numSegments: int
        The number of linear segments to have in this relationship.
    relation: str (optional, Default = "==")
        The type of relationship you'd like this to be: "==", "<=", or ">="
    args: list (optional, Default = ())
        A list of additional arguments to pass to "func"
    kwargs: dict (optional, Default = {})
        A dict of additional key-word arguments to pass to "func"
    verify: bool (optional, Default = True)
        An indication of whether or not you'd like to verify that the PWL approximation is a decent approximation by visually inspecting it.
    formulation: str (optional, Default = "DCC)
        The PWL formulation style to use. For options, see the "pw_repn" section of the "Piecewise Linear Expressions" sections of this site: https://pyomo.readthedocs.io/en/stable/pyomo_modeling_components/Expressions.html#piecewise-linear-expressions . Note that "SOS2" is generally the fastest, but it is only compatible with a few solvers ("guorbi","cplex", "cbc","scip")
    numFineResolutionSegments: int (optional, Default=50)
        When determining the optimal PWL points, we first approximate the "func" provided as a series of discrete points. This is the number of points to use in this discretization. Note that when numSegments gets bigger (e.g. larger than 6) it can take this code a long time to optimize the PWL points. In such a case, you may want to consider decreeing this value to boost computational time. Also note that if you bypass the automatic generation of the PWL points by passing an array of existing points as "func", you can simply disregard this parameter.

    Returns
    -------
    None
    """
    if isfunction(func):
        xData = np.linspace(xBounds[0], xBounds[1], numFineResolutionSegments)
        yData = np.array([func(x, *args, **kwargs) for x in xData])

        my_pwlf = PiecewiseLinFit(xData, yData)
        xBreaks = my_pwlf.fit(numSegments)
        yBreaks = my_pwlf.predict(xBreaks)

        if verify:
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.plot(xData, yData, label="Original Function")
            ax.plot(xBreaks, yBreaks, label="PWL Approximation")
            ax.legend()
            ax.set_title(
                'Your code will resume once you close this window.\nTo disable visual verification, pass "verify=False" to the PWL function.'
            )
            plt.show()
    else:
        xBreaks = func[:, 0]
        yBreaks = func[:, 1]
        if verify:
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.plot(xBreaks, yBreaks, ".", color="tab:blue")
            ax.plot(xBreaks, yBreaks, color="tab:blue")
            ax.set_title(
                'Your code will resume once you close this window.\nTo disable visual verification, pass "verify=False" to the PWL function.'
            )
            plt.show()

    constrTypeMap = {"==": "EQ", "<=": "UB", ">=": "LB"}
    if relation not in constrTypeMap:
        raise ValueError(f'"{relation}" is not a recognized relation.')
    relation = constrTypeMap[relation]

    relationshipName = f"{xVar}_{yVar}_PWL"

    setattr(
        model,
        relationshipName,
        Piecewise(
            yVar,
            xVar,
            pw_constr_type=relation,
            pw_pts=list(xBreaks),
            f_rule=list(yBreaks),
            pw_repn=formulation,
        ),
    )

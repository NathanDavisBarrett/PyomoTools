import pyomo.environ as pyo
import numpy as np
from scipy.optimize import linprog

from ..VectorRepresentation import VectorRepresentation
from ....base.Solvers import DefaultSolver


def PlotProb(A, b, c, xfeas=None):
    import matplotlib.pyplot as plt

    # Create a grid of points
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)
    Z = c[0] * X + c[1] * Y

    masks = [A[i, 0] * X + A[i, 1] * Y <= b[i] for i in range(A.shape[0])]
    mask = masks[0]
    for i in range(1, len(masks)):
        mask = mask & masks[i]

    # Plot the feasible region
    fig, ax = plt.subplots(1, 1)

    plt.imshow(
        mask.astype(int), extent=(-2, 2, -2, 2), origin="lower", cmap="Greys", alpha=0.3
    )

    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=True, fontsize=10)

    # Plot the lines corresponding to the inequalities
    for i in range(A.shape[0]):
        ax.plot(x, (b[i] - A[i, 0] * x) / A[i, 1], color="k")

    # Set plot limits and labels
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    if xfeas is not None:
        ax.scatter(
            [
                xfeas[0],
            ],
            [
                xfeas[1],
            ],
        )

    # Show the plot
    plt.show()


def executeModelTest(model):
    vecRep = VectorRepresentation(model)

    solver = DefaultSolver("LP")
    solver.solve(model)

    xOrig = [pyo.value(v) for v in vecRep.VAR_VEC]

    A, b, c, d, inequalityIndices, equalityIndices = (
        vecRep.Generate_Matrix_Representation()
    )

    A = A.toarray()
    print(A, inequalityIndices)

    A_ub = A[inequalityIndices, :]
    b_ub = b[inequalityIndices]
    A_eq = A[equalityIndices, :]
    b_eq = b[equalityIndices]

    hasEq = sum(A_eq.shape) > 0
    hasLeq = sum(A_ub.shape) > 0
    if hasEq and hasLeq:
        res = linprog(c, A_ub, b_ub, A_eq, b_eq)
    elif hasLeq:
        res = linprog(c, A_ub, b_ub)
    elif hasEq:
        res = linprog(c, A_eq=A_eq, b_eq=b_eq)
    else:
        raise Exception("No equality constraints or inequality constraints found.")

    assert res.success

    x = res.x

    assert np.allclose(x, xOrig)


def test_2Var4Constr():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(0, 1))
    model.y = pyo.Var(domain=pyo.NonNegativeReals)

    model.c = pyo.Constraint(expr=model.y == model.x)

    model.obj = pyo.Objective(expr=model.x, sense=pyo.maximize)

    executeModelTest(model)


def test_BigRandom():
    np.random.seed(0)
    n = 2  # vars
    m = 5  # constrs

    lower_bound = -1
    upper_bound = 1

    c = np.random.rand(n)

    # Generate a random feasible solution within the bounds
    x_feasible = np.random.uniform(lower_bound, upper_bound, n)

    # Generate a random constraint matrix
    A = np.random.rand(m, n)

    # Compute the right-hand side vector to ensure feasibility
    b = A @ x_feasible + np.random.rand(m)

    # Add bounds to the constraints
    A_bounds = np.vstack([np.eye(n), -np.eye(n)])
    b_bounds = np.hstack([upper_bound * np.ones(n), -lower_bound * np.ones(n)])

    A = np.vstack([A, A_bounds])
    b = np.hstack([b, b_bounds])

    m, n = A.shape

    model = pyo.ConcreteModel()
    model.N = pyo.Set(initialize=list(range(n)))
    model.M = pyo.Set(initialize=list(range(m)))
    model.x = pyo.Var(model.N)

    def constr(model, i):
        return sum(A[i, j] * model.x[j] for j in model.N) <= b[i]

    model.c = pyo.Constraint(model.M, rule=constr)

    model.obj = pyo.Objective(
        expr=sum(c[j] * model.x[j] for j in model.N), sense=pyo.minimize
    )

    vecRep = VectorRepresentation(model)

    Anew, bnew, cnew, dnew, inequalityIndices, equalityIndices = (
        vecRep.Generate_Matrix_Representation()
    )

    Anew = Anew.toarray()

    A_ub = Anew[inequalityIndices, :]
    b_ub = bnew[inequalityIndices]

    assert np.allclose(A, A_ub)
    assert np.allclose(b, b_ub)
    assert np.allclose(c, cnew)
    assert np.allclose(
        [
            dnew,
        ],
        [
            0,
        ],
    )

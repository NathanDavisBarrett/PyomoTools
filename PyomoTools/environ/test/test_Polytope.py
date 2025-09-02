import pyomo.environ as pyo

from ..Polytope import Polytope


def test_2D(plot=False):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(-5, 5))
    model.y = pyo.Var(bounds=(-5, 5))
    model.z = pyo.Var(bounds=(-5, 5))
    model.a = pyo.Var(bounds=(-100, 100))

    model.c = pyo.Constraint(
        expr=3 * (5 * model.x + model.y - 13) <= model.z / 2 + 10 + model.a
    )

    polytope = Polytope(model, [model.x, model.y])
    if plot:
        polytope.Plot()


def test_3D(plot=False):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(-5, 5))
    model.y = pyo.Var(bounds=(-5, 5))
    model.z = pyo.Var(bounds=(-5, 5))
    model.a = pyo.Var(bounds=(-100, 100))

    model.c = pyo.Constraint(
        expr=3 * (5 * model.x + model.y - 13) <= model.z / 2 + 10 + model.a
    )

    polytope = Polytope(model, [model.x, model.y, model.z])
    if plot:
        polytope.Plot()


def test_2D_DropConstr(plot=False):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(-5, 5))
    model.y = pyo.Var(bounds=(-5, 5))
    model.z = pyo.Var(bounds=(-5, 5))
    model.a = pyo.Var(bounds=(-100, 100))

    model.c = pyo.Constraint(
        expr=3 * (5 * model.x + model.y - 13) <= model.z / 2 + 10 + model.a
    )

    model.c2 = pyo.Constraint(expr=model.z + model.a == 1)

    polytope = Polytope(model, [model.x, model.y])
    if plot:
        polytope.Plot()

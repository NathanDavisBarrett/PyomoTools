import pyomo.kernel as pmo
import pytest

from ..Generate_MIP_Duals import Generate_MIP_Duals
from ....base.Solvers import DefaultSolver


def _ensure_lp_solver_available() -> None:
    solver = DefaultSolver("LP")
    if not solver.available(False):
        pytest.skip("LP solver not available for Generate_MIP_Duals tests")


def _build_simple_mip():
    model = pmo.block()
    model.x = pmo.variable(domain=pmo.NonNegativeReals)
    model.b = pmo.variable(domain=pmo.Binary)

    model.c1 = pmo.constraint(expr=model.x >= 2 * model.b)
    model.c2 = pmo.constraint(expr=model.x <= 5)
    model.obj = pmo.objective(expr=model.x)

    # Provide incumbent value for integer var so it can be fixed in the LP clone
    model.b.value = 1
    return model


def _build_nested_mip():
    model = pmo.block()
    model.y = pmo.variable(domain=pmo.NonNegativeReals)
    model.childBlock = pmo.block()
    model.childBlock.z = pmo.variable(domain=pmo.Binary)

    model.c_parent = pmo.constraint(expr=model.y >= 1)
    model.childBlock.c_child = pmo.constraint(expr=model.childBlock.z + model.y <= 3)
    model.obj = pmo.objective(expr=model.y + model.childBlock.z)

    model.childBlock.z.value = 1
    return model


def test_generate_mip_duals_populates_duals():
    _ensure_lp_solver_available()
    model = _build_simple_mip()

    Generate_MIP_Duals(model)
    print(list(model.dual.keys()))
    assert model.c1 in model.dual
    assert model.c2 in model.dual
    # Original integer variable should remain unfixed on the original model
    assert not model.b.fixed


def test_generate_mip_duals_adds_dual_suffix():
    _ensure_lp_solver_available()
    model = _build_simple_mip()

    # Ensure no dual suffix pre-exists
    if hasattr(model, "dual"):
        delattr(model, "dual")

    Generate_MIP_Duals(model)

    assert hasattr(model, "dual")
    assert model.c1 in model.dual


def test_generate_mip_duals_propagates_to_child_blocks():
    _ensure_lp_solver_available()
    model = _build_nested_mip()

    Generate_MIP_Duals(model)

    # Duals are present for both parent and child constraints
    assert model.c_parent in model.dual
    assert model.childBlock.c_child in model.dual
    # Child block also gains its own dual suffix entry
    assert hasattr(model.childBlock, "dual")
    assert model.childBlock.c_child in model.childBlock.dual

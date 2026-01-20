import pytest
import pyomo.kernel as pmo
from ..IncorporateDuals import IncorporateDuals
from ....base.Solvers import DefaultSolver


@pytest.fixture
def simple_model():
    """Create a simple model with constraints for testing."""
    model = pmo.block()
    model.x = pmo.variable()
    model.c1 = pmo.constraint(expr=model.x >= 0)
    model.c2 = pmo.constraint(expr=model.x <= 10)
    return model


@pytest.fixture
def nested_model():
    """Create a nested model with child blocks."""
    model = pmo.block()
    model.x = pmo.variable()
    model.c1 = pmo.constraint(expr=model.x >= 0)

    model.childBlock = pmo.block()
    model.childBlock.y = pmo.variable()
    model.childBlock.c2 = pmo.constraint(expr=model.childBlock.y >= 5)

    return model


def test_incorporate_duals_simple(simple_model):
    """Test incorporating duals into a simple model."""
    simple_model.obj = pmo.objective(expr=simple_model.x)
    simple_model.dual = pmo.suffix(direction=pmo.suffix.IMPORT)

    solver = DefaultSolver("LP")
    solver.solve(simple_model)

    IncorporateDuals(simple_model)

    # After solving, check that duals exist and are accessible
    assert simple_model.c1 in simple_model.dual
    assert simple_model.c2 in simple_model.dual


def test_incorporate_duals_nested(nested_model):
    """Test incorporating duals into nested blocks."""
    nested_model.obj = pmo.objective(expr=nested_model.x + nested_model.childBlock.y)
    nested_model.dual = pmo.suffix(direction=pmo.suffix.IMPORT)

    solver = DefaultSolver("LP")
    solver.solve(nested_model)

    IncorporateDuals(nested_model)

    # After solving, check that duals exist in both parent and child blocks
    assert nested_model.c1 in nested_model.dual
    assert nested_model.childBlock.c2 in nested_model.dual


def test_no_dual_suffix_raises_error(simple_model):
    """Test that missing dual suffix raises ValueError."""
    with pytest.raises(ValueError, match="does not have a dual suffix"):
        IncorporateDuals(simple_model)


def test_constraint_list(simple_model):
    """Test incorporating duals with constraint_list."""
    simple_model.constr_list = pmo.constraint_list()
    simple_model.constr_list.append(pmo.constraint(simple_model.x >= 1))
    simple_model.constr_list.append(pmo.constraint(simple_model.x <= 9))

    simple_model.obj = pmo.objective(expr=simple_model.x)
    simple_model.dual = pmo.suffix(direction=pmo.suffix.IMPORT)

    solver = DefaultSolver("LP")
    solver.solve(simple_model)

    IncorporateDuals(simple_model)

    assert isinstance(simple_model.dual[simple_model.constr_list[0]], (int, float))
    assert isinstance(simple_model.dual[simple_model.constr_list[1]], (int, float))


def test_constraint_dict(simple_model):
    """Test incorporating duals with constraint_dict."""
    simple_model.constr_dict = pmo.constraint_dict()
    simple_model.constr_dict["c1"] = pmo.constraint(expr=simple_model.x >= 1)
    simple_model.constr_dict["c2"] = pmo.constraint(expr=simple_model.x <= 9)

    simple_model.obj = pmo.objective(expr=simple_model.x)
    simple_model.dual = pmo.suffix(direction=pmo.suffix.IMPORT)

    solver = DefaultSolver("LP")
    solver.solve(simple_model)

    IncorporateDuals(simple_model)

    assert simple_model.constr_dict["c1"] in simple_model.dual
    assert simple_model.constr_dict["c2"] in simple_model.dual


def test_block_list():
    """Test incorporating duals with block_list."""
    model = pmo.block()
    model.block_list = pmo.block_list()

    model.block_list.append(pmo.block())
    b1 = model.block_list[0]
    b1.x = pmo.variable()
    b1.c = pmo.constraint(expr=b1.x >= 0)

    model.obj = pmo.objective(expr=b1.x)
    model.dual = pmo.suffix(direction=pmo.suffix.IMPORT)

    solver = DefaultSolver("LP")
    solver.solve(model)

    IncorporateDuals(model)

    assert b1.c in model.dual


def test_missing_dual_warning(simple_model):
    """Test that missing dual values trigger a warning."""
    simple_model.obj = pmo.objective(expr=simple_model.x)
    simple_model.dual = pmo.suffix(direction=pmo.suffix.IMPORT)

    # Create a new constraint that won't have a dual value
    simple_model.c3 = pmo.constraint(expr=simple_model.x >= -5)

    with pytest.warns(UserWarning, match="Dual value for constraint"):
        IncorporateDuals(simple_model)

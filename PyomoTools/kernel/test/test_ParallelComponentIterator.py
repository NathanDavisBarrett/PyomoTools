import pyomo.kernel as pmo
import pytest

from ..ParallelComponentIterator import ParallelComponentIterator


def create_simple_models_with_vars():
    """Create two simple models with individual variables."""
    model1 = pmo.block()
    model1.x = pmo.variable()
    model1.y = pmo.variable()
    model1.x.value = 1.0
    model1.y.value = 2.0

    model2 = pmo.block()
    model2.x = pmo.variable()
    model2.y = pmo.variable()
    model2.x.value = 3.0
    model2.y.value = 4.0

    return [model1, model2]


def create_models_with_constraints():
    """Create two models with variables and constraints."""
    model1 = pmo.block()
    model1.x = pmo.variable()
    model1.y = pmo.variable()
    model1.c1 = pmo.constraint(model1.x + model1.y <= 10.0)
    model1.c2 = pmo.constraint(model1.x >= 0.0)

    model2 = pmo.block()
    model2.x = pmo.variable()
    model2.y = pmo.variable()
    model2.c1 = pmo.constraint(model2.x + model2.y <= 20.0)
    model2.c2 = pmo.constraint(model2.x >= 5.0)

    return [model1, model2]


def create_models_with_objectives():
    """Create two models with variables and objectives."""
    model1 = pmo.block()
    model1.x = pmo.variable()
    model1.y = pmo.variable()
    model1.obj = pmo.objective(model1.x + model1.y)

    model2 = pmo.block()
    model2.x = pmo.variable()
    model2.y = pmo.variable()
    model2.obj = pmo.objective(model2.x + model2.y)

    return [model1, model2]


def create_models_with_variable_lists():
    """Create two models with variable lists."""
    model1 = pmo.block()
    model1.x_list = pmo.variable_list([pmo.variable() for _ in range(3)])
    model1.y = pmo.variable()

    model2 = pmo.block()
    model2.x_list = pmo.variable_list([pmo.variable() for _ in range(3)])
    model2.y = pmo.variable()

    return [model1, model2]


def create_models_with_variable_dicts():
    """Create two models with variable dicts."""
    model1 = pmo.block()
    model1.x_dict = pmo.variable_dict()
    model1.x_dict["a"] = pmo.variable()
    model1.x_dict["b"] = pmo.variable()
    model1.x_dict["c"] = pmo.variable()

    model2 = pmo.block()
    model2.x_dict = pmo.variable_dict()
    model2.x_dict["a"] = pmo.variable()
    model2.x_dict["b"] = pmo.variable()
    model2.x_dict["c"] = pmo.variable()

    return [model1, model2]


def create_models_with_constraint_lists():
    """Create two models with constraint lists."""
    model1 = pmo.block()
    model1.x = pmo.variable_list([pmo.variable() for _ in range(3)])
    model1.c_list = pmo.constraint_list(
        [
            pmo.constraint(model1.x[0] <= 10.0),
            pmo.constraint(model1.x[1] >= 0.0),
            pmo.constraint(model1.x[2] == 5.0),
        ]
    )

    model2 = pmo.block()
    model2.x = pmo.variable_list([pmo.variable() for _ in range(3)])
    model2.c_list = pmo.constraint_list(
        [
            pmo.constraint(model2.x[0] <= 20.0),
            pmo.constraint(model2.x[1] >= 5.0),
            pmo.constraint(model2.x[2] == 10.0),
        ]
    )

    return [model1, model2]


def create_models_with_constraint_dicts():
    """Create two models with constraint dicts."""
    model1 = pmo.block()
    model1.x = pmo.variable()
    model1.y = pmo.variable()
    model1.c_dict = pmo.constraint_dict()
    model1.c_dict["lower"] = pmo.constraint(model1.x >= 0.0)
    model1.c_dict["upper"] = pmo.constraint(model1.x <= 10.0)
    model1.c_dict["relation"] = pmo.constraint(model1.x == 2 * model1.y)

    model2 = pmo.block()
    model2.x = pmo.variable()
    model2.y = pmo.variable()
    model2.c_dict = pmo.constraint_dict()
    model2.c_dict["lower"] = pmo.constraint(model2.x >= 5.0)
    model2.c_dict["upper"] = pmo.constraint(model2.x <= 20.0)
    model2.c_dict["relation"] = pmo.constraint(model2.x == 3 * model2.y)

    return [model1, model2]


def create_hierarchical_models():
    """Create two hierarchical models with sub-blocks."""
    model1 = pmo.block()
    model1.x = pmo.variable()
    model1.y = pmo.variable()
    model1.c1 = pmo.constraint(model1.x + model1.y <= 10.0)

    # Sub-block 1
    model1.sub1 = pmo.block()
    model1.sub1.a = pmo.variable()
    model1.sub1.b = pmo.variable()
    model1.sub1.c_sub = pmo.constraint(model1.sub1.a + model1.sub1.b >= 5.0)

    # Sub-block 2
    model1.sub2 = pmo.block()
    model1.sub2.z = pmo.variable()
    model1.sub2.c_sub = pmo.constraint(model1.sub2.z == 3.0)

    model2 = pmo.block()
    model2.x = pmo.variable()
    model2.y = pmo.variable()
    model2.c1 = pmo.constraint(model2.x + model2.y <= 20.0)

    # Sub-block 1
    model2.sub1 = pmo.block()
    model2.sub1.a = pmo.variable()
    model2.sub1.b = pmo.variable()
    model2.sub1.c_sub = pmo.constraint(model2.sub1.a + model2.sub1.b >= 10.0)

    # Sub-block 2
    model2.sub2 = pmo.block()
    model2.sub2.z = pmo.variable()
    model2.sub2.c_sub = pmo.constraint(model2.sub2.z == 6.0)

    return [model1, model2]


def create_deeply_nested_models():
    """Create models with multiple levels of nesting."""
    model1 = pmo.block()
    model1.x = pmo.variable()

    model1.level1 = pmo.block()
    model1.level1.y = pmo.variable()

    model1.level1.level2 = pmo.block()
    model1.level1.level2.z = pmo.variable()

    model1.level1.level2.level3 = pmo.block()
    model1.level1.level2.level3.w = pmo.variable()

    model2 = pmo.block()
    model2.x = pmo.variable()

    model2.level1 = pmo.block()
    model2.level1.y = pmo.variable()

    model2.level1.level2 = pmo.block()
    model2.level1.level2.z = pmo.variable()

    model2.level1.level2.level3 = pmo.block()
    model2.level1.level2.level3.w = pmo.variable()

    return [model1, model2]


def create_models_with_block_lists():
    """Create models with block lists."""
    model1 = pmo.block()
    model1.x = pmo.variable()
    model1.blocks = pmo.block_list()
    for i in range(3):
        b = pmo.block()
        b.var = pmo.variable()
        b.c = pmo.constraint(b.var <= i + 1.0)
        model1.blocks.append(b)

    model2 = pmo.block()
    model2.x = pmo.variable()
    model2.blocks = pmo.block_list()
    for i in range(3):
        b = pmo.block()
        b.var = pmo.variable()
        b.c = pmo.constraint(b.var <= (i + 1.0) * 2)
        model2.blocks.append(b)

    return [model1, model2]


def create_models_with_block_dicts():
    """Create models with block dicts."""
    model1 = pmo.block()
    model1.x = pmo.variable()
    model1.blocks = pmo.block_dict()
    for key in ["first", "second", "third"]:
        b = pmo.block()
        b.var = pmo.variable()
        b.c = pmo.constraint(b.var >= 0.0)
        model1.blocks[key] = b

    model2 = pmo.block()
    model2.x = pmo.variable()
    model2.blocks = pmo.block_dict()
    for key in ["first", "second", "third"]:
        b = pmo.block()
        b.var = pmo.variable()
        b.c = pmo.constraint(b.var >= 5.0)
        model2.blocks[key] = b

    return [model1, model2]


def create_comprehensive_models():
    """Create comprehensive models with all component types."""
    model1 = pmo.block()

    # Individual components
    model1.x = pmo.variable()
    model1.y = pmo.variable()
    model1.c1 = pmo.constraint(model1.x + model1.y <= 10.0)
    model1.obj = pmo.objective(model1.x + model1.y)

    # Lists
    model1.var_list = pmo.variable_list([pmo.variable() for _ in range(2)])
    model1.constr_list = pmo.constraint_list(
        [
            pmo.constraint(model1.var_list[0] >= 0.0),
            pmo.constraint(model1.var_list[1] <= 5.0),
        ]
    )

    # Dicts
    model1.var_dict = pmo.variable_dict()
    model1.var_dict["a"] = pmo.variable()
    model1.var_dict["b"] = pmo.variable()

    # Sub-block
    model1.sub = pmo.block()
    model1.sub.z = pmo.variable()
    model1.sub.c_sub = pmo.constraint(model1.sub.z == 3.0)

    model2 = pmo.block()

    # Individual components
    model2.x = pmo.variable()
    model2.y = pmo.variable()
    model2.c1 = pmo.constraint(model2.x + model2.y <= 20.0)
    model2.obj = pmo.objective(model2.x + model2.y)

    # Lists
    model2.var_list = pmo.variable_list([pmo.variable() for _ in range(2)])
    model2.constr_list = pmo.constraint_list(
        [
            pmo.constraint(model2.var_list[0] >= 5.0),
            pmo.constraint(model2.var_list[1] <= 10.0),
        ]
    )

    # Dicts
    model2.var_dict = pmo.variable_dict()
    model2.var_dict["a"] = pmo.variable()
    model2.var_dict["b"] = pmo.variable()

    # Sub-block
    model2.sub = pmo.block()
    model2.sub.z = pmo.variable()
    model2.sub.c_sub = pmo.constraint(model2.sub.z == 6.0)

    return [model1, model2]


# Tests


def test_simple_variables():
    """Test iteration over simple models with only variables."""
    models = create_simple_models_with_vars()
    iterator = ParallelComponentIterator(
        models, collect_vars=True, collect_constrs=False, collect_objs=False
    )

    components = list(iterator)
    assert len(components) == 2  # x and y
    assert all(len(comp) == 2 for comp in components)  # Two models per component

    # Check that we get the right variables
    names = [comp[0].name for comp in components]
    assert set(names) == {"x", "y"}


def test_variables_and_constraints():
    """Test iteration over models with variables and constraints."""
    models = create_models_with_constraints()
    iterator = ParallelComponentIterator(
        models, collect_vars=True, collect_constrs=True, collect_objs=False
    )

    components = list(iterator)
    # 2 variables + 2 constraints = 4 components
    assert len(components) == 4

    var_names = set()
    constr_names = set()
    for comp in components:
        if isinstance(comp[0], pmo.variable):
            var_names.add(comp[0].name)
        elif isinstance(comp[0], pmo.constraint):
            constr_names.add(comp[0].name)

    assert var_names == {"x", "y"}
    assert constr_names == {"c1", "c2"}


def test_objectives():
    """Test iteration over models with objectives."""
    models = create_models_with_objectives()
    iterator = ParallelComponentIterator(
        models, collect_vars=True, collect_constrs=False, collect_objs=True
    )

    components = list(iterator)
    # 2 variables + 1 objective = 3 components
    assert len(components) == 3

    obj_count = sum(1 for comp in components if isinstance(comp[0], pmo.objective))
    assert obj_count == 1


def test_variable_lists():
    """Test iteration over models with variable lists."""
    models = create_models_with_variable_lists()
    iterator = ParallelComponentIterator(
        models, collect_vars=True, collect_constrs=False, collect_objs=False
    )

    components = list(iterator)
    # 3 from x_list + 1 y = 4 variables
    assert len(components) == 4

    # All should be variables
    assert all(isinstance(comp[0], pmo.variable) for comp in components)


def test_variable_dicts():
    """Test iteration over models with variable dicts."""
    models = create_models_with_variable_dicts()
    iterator = ParallelComponentIterator(
        models, collect_vars=True, collect_constrs=False, collect_objs=False
    )

    components = list(iterator)
    # 3 variables in dict
    assert len(components) == 3

    # All should be variables
    assert all(isinstance(comp[0], pmo.variable) for comp in components)


def test_constraint_lists():
    """Test iteration over models with constraint lists."""
    models = create_models_with_constraint_lists()
    iterator = ParallelComponentIterator(
        models, collect_vars=True, collect_constrs=True, collect_objs=False
    )

    components = list(iterator)
    # 3 from x variable list + 3 from c_list = 6 components
    assert len(components) == 6

    constr_count = sum(1 for comp in components if isinstance(comp[0], pmo.constraint))
    assert constr_count == 3


def test_constraint_dicts():
    """Test iteration over models with constraint dicts."""
    models = create_models_with_constraint_dicts()
    iterator = ParallelComponentIterator(
        models, collect_vars=True, collect_constrs=True, collect_objs=False
    )

    components = list(iterator)
    # 2 variables + 3 constraints in dict = 5 components
    assert len(components) == 5

    constr_count = sum(1 for comp in components if isinstance(comp[0], pmo.constraint))
    assert constr_count == 3


def test_hierarchical_models():
    """Test iteration over hierarchical models with sub-blocks."""
    models = create_hierarchical_models()
    iterator = ParallelComponentIterator(
        models, collect_vars=True, collect_constrs=True, collect_objs=False
    )

    components = list(iterator)

    # Root level: 2 vars + 1 constraint
    # sub1: 2 vars + 1 constraint
    # sub2: 1 var + 1 constraint
    # Total: 5 vars + 3 constraints = 8 components
    assert len(components) == 8

    var_count = sum(1 for comp in components if isinstance(comp[0], pmo.variable))
    constr_count = sum(1 for comp in components if isinstance(comp[0], pmo.constraint))

    assert var_count == 5
    assert constr_count == 3


def test_deeply_nested_models():
    """Test iteration over deeply nested models."""
    models = create_deeply_nested_models()
    iterator = ParallelComponentIterator(
        models, collect_vars=True, collect_constrs=False, collect_objs=False
    )

    components = list(iterator)
    # x, level1.y, level1.level2.z, level1.level2.level3.w = 4 variables
    assert len(components) == 4

    # All should be variables
    assert all(isinstance(comp[0], pmo.variable) for comp in components)


def test_block_lists():
    """Test iteration over models with block lists."""
    models = create_models_with_block_lists()
    iterator = ParallelComponentIterator(
        models, collect_vars=True, collect_constrs=True, collect_objs=False
    )

    components = list(iterator)
    # 1 root var + 3 blocks * (1 var + 1 constraint) = 1 + 6 = 7 components
    assert len(components) == 7

    var_count = sum(1 for comp in components if isinstance(comp[0], pmo.variable))
    constr_count = sum(1 for comp in components if isinstance(comp[0], pmo.constraint))

    assert var_count == 4
    assert constr_count == 3


def test_block_dicts():
    """Test iteration over models with block dicts."""
    models = create_models_with_block_dicts()
    iterator = ParallelComponentIterator(
        models, collect_vars=True, collect_constrs=True, collect_objs=False
    )

    components = list(iterator)
    # 1 root var + 3 blocks * (1 var + 1 constraint) = 1 + 6 = 7 components
    assert len(components) == 7

    var_count = sum(1 for comp in components if isinstance(comp[0], pmo.variable))
    constr_count = sum(1 for comp in components if isinstance(comp[0], pmo.constraint))

    assert var_count == 4
    assert constr_count == 3


def test_comprehensive_models():
    """Test iteration over comprehensive models with all component types."""
    models = create_comprehensive_models()
    iterator = ParallelComponentIterator(
        models, collect_vars=True, collect_constrs=True, collect_objs=True
    )

    components = list(iterator)

    # Root level: 2 individual vars + 2 var_list + 2 var_dict = 6 vars
    #             1 constraint + 2 constr_list = 3 constraints
    #             1 objective
    # sub block: 1 var + 1 constraint
    # Total: 7 vars + 4 constraints + 1 objective = 12 components
    assert len(components) == 12

    var_count = sum(1 for comp in components if isinstance(comp[0], pmo.variable))
    constr_count = sum(1 for comp in components if isinstance(comp[0], pmo.constraint))
    obj_count = sum(1 for comp in components if isinstance(comp[0], pmo.objective))

    assert var_count == 7
    assert constr_count == 4
    assert obj_count == 1


def test_parallel_values():
    """Test that parallel iteration correctly pairs components from different models."""
    models = create_simple_models_with_vars()
    models[0].x.value = 10.0
    models[1].x.value = 20.0

    iterator = ParallelComponentIterator(
        models, collect_vars=True, collect_constrs=False, collect_objs=False
    )

    for components in iterator:
        if components[0].name == "x":
            assert components[0].value == 10.0
            assert components[1].value == 20.0


def test_three_models():
    """Test iteration with three models instead of two."""
    model3 = pmo.block()
    model3.x = pmo.variable()
    model3.y = pmo.variable()
    model3.x.value = 5.0
    model3.y.value = 6.0

    models = create_simple_models_with_vars()
    models.append(model3)

    iterator = ParallelComponentIterator(
        models, collect_vars=True, collect_constrs=False, collect_objs=False
    )

    components = list(iterator)
    assert len(components) == 2  # x and y
    assert all(len(comp) == 3 for comp in components)  # Three models per component


def test_collect_only_constraints():
    """Test with only collect_constrs=True."""
    models = create_models_with_constraints()
    iterator = ParallelComponentIterator(
        models, collect_vars=False, collect_constrs=True, collect_objs=False
    )

    components = list(iterator)
    # Only 2 constraints (c1 and c2), no variables
    assert len(components) == 2
    assert all(isinstance(comp[0], pmo.constraint) for comp in components)


def test_empty_models():
    """Test with empty models."""
    model1 = pmo.block()
    model2 = pmo.block()

    iterator = ParallelComponentIterator(
        [model1, model2],
        collect_vars=True,
        collect_constrs=True,
        collect_objs=True,
    )

    components = list(iterator)
    assert len(components) == 0


def test_mismatched_variable_names():
    """Test that validation catches mismatched variable names."""
    model1 = pmo.block()
    model1.x = pmo.variable()
    model1.y = pmo.variable()

    model2 = pmo.block()
    model2.x = pmo.variable()
    model2.z = pmo.variable()  # Different name

    with pytest.raises(
        ValueError, match="do not have the same set of individual variables"
    ):
        ParallelComponentIterator(
            [model1, model2],
            collect_vars=True,
            collect_constrs=False,
            collect_objs=False,
        )


def test_mismatched_list_lengths():
    """Test that validation catches mismatched list lengths."""
    model1 = pmo.block()
    model1.x_list = pmo.variable_list([pmo.variable() for _ in range(3)])

    model2 = pmo.block()
    model2.x_list = pmo.variable_list(
        [pmo.variable() for _ in range(5)]
    )  # Different length

    with pytest.raises(
        ValueError, match="do not have the same sizes for variable list/tuple"
    ):
        ParallelComponentIterator(
            [model1, model2],
            collect_vars=True,
            collect_constrs=False,
            collect_objs=False,
        )


def test_mismatched_dict_keys():
    """Test that validation catches mismatched dict keys."""
    model1 = pmo.block()
    model1.x_dict = pmo.variable_dict()
    model1.x_dict["a"] = pmo.variable()
    model1.x_dict["b"] = pmo.variable()

    model2 = pmo.block()
    model2.x_dict = pmo.variable_dict()
    model2.x_dict["a"] = pmo.variable()
    model2.x_dict["c"] = pmo.variable()  # Different key

    with pytest.raises(ValueError, match="do not have the same keys for variable dict"):
        ParallelComponentIterator(
            [model1, model2],
            collect_vars=True,
            collect_constrs=False,
            collect_objs=False,
        )


def test_mismatched_constraint_structure():
    """Test that validation catches mismatched constraint structures."""
    model1 = pmo.block()
    model1.x = pmo.variable()
    model1.c1 = pmo.constraint(model1.x <= 10.0)

    model2 = pmo.block()
    model2.x = pmo.variable()
    model2.c2 = pmo.constraint(model2.x <= 10.0)  # Different name

    with pytest.raises(
        ValueError, match="do not have the same set of individual constraints"
    ):
        ParallelComponentIterator(
            [model1, model2],
            collect_vars=True,
            collect_constrs=True,
            collect_objs=False,
        )


def test_mismatched_block_structure():
    """Test that validation catches mismatched block structures."""
    model1 = pmo.block()
    model1.x = pmo.variable()
    model1.sub1 = pmo.block()
    model1.sub1.y = pmo.variable()

    model2 = pmo.block()
    model2.x = pmo.variable()
    model2.sub2 = pmo.block()  # Different block name
    model2.sub2.y = pmo.variable()

    with pytest.raises(
        ValueError, match="do not have the same set of individual blocks"
    ):
        ParallelComponentIterator(
            [model1, model2],
            collect_vars=True,
            collect_constrs=False,
            collect_objs=False,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

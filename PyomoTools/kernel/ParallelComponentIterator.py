import pyomo.kernel as pmo

from typing import List


class ParallelComponentIterator:
    def __init__(
        self,
        models: List[pmo.block],
        collect_vars: bool,
        collect_constrs: bool,
        collect_objs: bool,
    ):
        """
        An iterator to iterate over the elements of multiple duplicate Pyomo models in parallel.

        Parameters
        ----------
        models : list of pmo.block
            The Pyomo models to iterate over in parallel.
        collect_vars : bool
            Whether to include variable objects in the iteration.
        collect_constrs : bool
            Whether to include constraint objects in the iteration.
        collect_objs : bool
            Whether to include objective objects in the iteration.
        """
        self.models = models
        self.collect_vars = collect_vars
        self.collect_constrs = collect_constrs
        self.collect_objs = collect_objs
        self._collect_names()
        self._validate()

    def _collect_names(self):
        if self.collect_vars:
            self.var_names = [set() for _ in self.models]
            self.var_list_names = [{} for _ in self.models]
            self.var_dict_names = [{} for _ in self.models]

        if self.collect_constrs:
            self.constr_names = [set() for _ in self.models]
            self.constr_list_names = [{} for _ in self.models]
            self.constr_dict_names = [{} for _ in self.models]

        if self.collect_objs:
            self.obj_names = [set() for _ in self.models]
            self.obj_list_names = [{} for _ in self.models]
            self.obj_dict_names = [{} for _ in self.models]

        self.block_names = [set() for _ in self.models]
        self.block_list_names = [{} for _ in self.models]
        self.block_dict_names = [{} for _ in self.models]

        for i, model in enumerate(self.models):
            for c in model.children():
                collected = False
                if self.collect_vars:
                    if isinstance(c, pmo.variable):
                        self.var_names[i].add(c.local_name)
                        collected = True
                    elif isinstance(c, (pmo.variable_list, pmo.variable_tuple)):
                        self.var_list_names[i][c.local_name] = len(c)
                        collected = True
                    elif isinstance(c, pmo.variable_dict):
                        self.var_dict_names[i][c.local_name] = set(c.keys())
                        collected = True
                if not collected and self.collect_constrs:
                    if isinstance(c, pmo.constraint):
                        self.constr_names[i].add(c.local_name)
                        collected = True
                    elif isinstance(c, (pmo.constraint_list, pmo.constraint_tuple)):
                        self.constr_list_names[i][c.local_name] = len(c)
                        collected = True
                    elif isinstance(c, pmo.constraint_dict):
                        self.constr_dict_names[i][c.local_name] = set(c.keys())
                        collected = True
                if not collected and self.collect_objs:
                    if isinstance(c, pmo.objective):
                        self.obj_names[i].add(c.local_name)
                        collected = True
                    elif isinstance(c, (pmo.objective_list, pmo.objective_tuple)):
                        self.obj_list_names[i][c.local_name] = len(c)
                        collected = True
                    elif isinstance(c, pmo.objective_dict):
                        self.obj_dict_names[i][c.local_name] = set(c.keys())
                        collected = True
                if not collected:
                    if isinstance(c, pmo.block):
                        self.block_names[i].add(c.local_name)
                    elif isinstance(c, (pmo.block_list, pmo.block_tuple)):
                        self.block_list_names[i][c.local_name] = len(c)
                    elif isinstance(c, pmo.block_dict):
                        self.block_dict_names[i][c.local_name] = set(c.keys())

    def _validate(self):
        for i in range(1, len(self.models)):
            if self.collect_vars:
                if self.var_names[i] != self.var_names[0]:
                    raise ValueError(
                        f"ParallelVariableIterator: The provided models do not have the same set of individual variables: Model #{i}: {self.var_names[i]} vs Model #0: {self.var_names[0]}."
                    )
                if self.var_list_names[i].keys() != self.var_list_names[0].keys():
                    raise ValueError(
                        f"ParallelVariableIterator: The provided models do not have the same set of variable lists/tuples: Model #{i}: {self.var_list_names[i].keys()} vs Model #0: {self.var_list_names[0].keys()}."
                    )
                for k in self.var_list_names[i]:
                    if self.var_list_names[i][k] != self.var_list_names[0][k]:
                        raise ValueError(
                            f'ParallelVariableIterator: The provided models do not have the same sizes for variable list/tuple "{k}": Model #{i}: {self.var_list_names[i][k]} vs Model #0: {self.var_list_names[0][k]}.'
                        )
                if self.var_dict_names[i].keys() != self.var_dict_names[0].keys():
                    raise ValueError(
                        f"ParallelVariableIterator: The provided models do not have the same set of variable dicts: Model #{i}: {self.var_dict_names[i].keys()} vs Model #0: {self.var_dict_names[0].keys()}."
                    )
                for k in self.var_dict_names[i]:
                    if self.var_dict_names[i][k] != self.var_dict_names[0][k]:
                        raise ValueError(
                            f'ParallelVariableIterator: The provided models do not have the same keys for variable dict "{k}": Model #{i}: {self.var_dict_names[i][k]} vs Model #0: {self.var_dict_names[0][k]}.'
                        )
            if self.collect_constrs:
                if self.constr_names[i] != self.constr_names[0]:
                    raise ValueError(
                        f"ParallelConstraintIterator: The provided models do not have the same set of individual constraints: Model #{i}: {self.constr_names[i]} vs Model #0: {self.constr_names[0]}."
                    )
                if self.constr_list_names[i].keys() != self.constr_list_names[0].keys():
                    raise ValueError(
                        f"ParallelConstraintIterator: The provided models do not have the same set of constraint lists/tuples: Model #{i}: {self.constr_list_names[i].keys()} vs Model #0: {self.constr_list_names[0].keys()}."
                    )
                for k in self.constr_list_names[i]:
                    if self.constr_list_names[i][k] != self.constr_list_names[0][k]:
                        raise ValueError(
                            f'ParallelConstraintIterator: The provided models do not have the same sizes for constraint list/tuple "{k}": Model #{i}: {self.constr_list_names[i][k]} vs Model #0: {self.constr_list_names[0][k]}.'
                        )
                if self.constr_dict_names[i].keys() != self.constr_dict_names[0].keys():
                    raise ValueError(
                        f"ParallelConstraintIterator: The provided models do not have the same set of constraint dicts: Model #{i}: {self.constr_dict_names[i].keys()} vs Model #0: {self.constr_dict_names[0].keys()}."
                    )
                for k in self.constr_dict_names[i]:
                    if self.constr_dict_names[i][k] != self.constr_dict_names[0][k]:
                        raise ValueError(
                            f'ParallelConstraintIterator: The provided models do not have the same keys for constraint dict "{k}": Model #{i}: {self.constr_dict_names[i][k]} vs Model #0: {self.constr_dict_names[0][k]}.'
                        )
            if self.collect_objs:
                if self.obj_names[i] != self.obj_names[0]:
                    raise ValueError(
                        f"ParallelObjectiveIterator: The provided models do not have the same set of individual objectives: Model #{i}: {self.obj_names[i]} vs Model #0: {self.obj_names[0]}."
                    )
                if self.obj_list_names[i].keys() != self.obj_list_names[0].keys():
                    raise ValueError(
                        f"ParallelObjectiveIterator: The provided models do not have the same set of objective lists/tuples: Model #{i}: {self.obj_list_names[i].keys()} vs Model #0: {self.obj_list_names[0].keys()}."
                    )
                for k in self.obj_list_names[i]:
                    if self.obj_list_names[i][k] != self.obj_list_names[0][k]:
                        raise ValueError(
                            f'ParallelObjectiveIterator: The provided models do not have the same sizes for objective list/tuple "{k}": Model #{i}: {self.obj_list_names[i][k]} vs Model #0: {self.obj_list_names[0][k]}.'
                        )
                if self.obj_dict_names[i].keys() != self.obj_dict_names[0].keys():
                    raise ValueError(
                        f"ParallelObjectiveIterator: The provided models do not have the same set of objective dicts: Model #{i}: {self.obj_dict_names[i].keys()} vs Model #0: {self.obj_dict_names[0].keys()}."
                    )
                for k in self.obj_dict_names[i]:
                    if self.obj_dict_names[i][k] != self.obj_dict_names[0][k]:
                        raise ValueError(
                            f'ParallelObjectiveIterator: The provided models do not have the same keys for objective dict "{k}": Model #{i}: {self.obj_dict_names[i][k]} vs Model #0: {self.obj_dict_names[0][k]}.'
                        )

            if self.block_names[i] != self.block_names[0]:
                raise ValueError(
                    f"ParallelBlockIterator: The provided models do not have the same set of individual blocks: Model #{i}: {self.block_names[i]} vs Model #0: {self.block_names[0]}."
                )
            if self.block_list_names[i].keys() != self.block_list_names[0].keys():
                raise ValueError(
                    f"ParallelBlockIterator: The provided models do not have the same set of block lists/tuples: Model #{i}: {self.block_list_names[i].keys()} vs Model #0: {self.block_list_names[0].keys()}."
                )
            for k in self.block_list_names[i]:
                if self.block_list_names[i][k] != self.block_list_names[0][k]:
                    raise ValueError(
                        f'ParallelBlockIterator: The provided models do not have the same sizes for block list/tuple "{k}": Model #{i}: {self.block_list_names[i][k]} vs Model #0: {self.block_list_names[0][k]}.'
                    )
            if self.block_dict_names[i].keys() != self.block_dict_names[0].keys():
                raise ValueError(
                    f"ParallelBlockIterator: The provided models do not have the same set of block dicts: Model #{i}: {self.block_dict_names[i].keys()} vs Model #0: {self.block_dict_names[0].keys()}."
                )
            for k in self.block_dict_names[i]:
                if self.block_dict_names[i][k] != self.block_dict_names[0][k]:
                    raise ValueError(
                        f'ParallelBlockIterator: The provided models do not have the same keys for block dict "{k}": Model #{i}: {self.block_dict_names[i][k]} vs Model #0: {self.block_dict_names[0][k]}.'
                    )

    def __iter__(self):
        if self.collect_vars:
            var_names = self.var_names[0]
            for name in var_names:
                yield [getattr(model, name) for model in self.models]
            var_list_names = self.var_list_names[0]
            for name in var_list_names:
                var_lists = [getattr(model, name) for model in self.models]
                for i in range(var_list_names[name]):
                    yield [var_list[i] for var_list in var_lists]
            var_dict_names = self.var_dict_names[0]
            for name in var_dict_names:
                var_dicts = [getattr(model, name) for model in self.models]
                for key in var_dict_names[name]:
                    yield [var_dict[key] for var_dict in var_dicts]
        if self.collect_constrs:
            constr_names = self.constr_names[0]
            for name in constr_names:
                yield [getattr(model, name) for model in self.models]
            constr_list_names = self.constr_list_names[0]
            for name in constr_list_names:
                constr_lists = [getattr(model, name) for model in self.models]
                for i in range(constr_list_names[name]):
                    yield [constr_list[i] for constr_list in constr_lists]
            constr_dict_names = self.constr_dict_names[0]
            for name in constr_dict_names:
                constr_dicts = [getattr(model, name) for model in self.models]
                for key in constr_dict_names[name]:
                    yield [constr_dict[key] for constr_dict in constr_dicts]
        if self.collect_objs:
            obj_names = self.obj_names[0]
            for name in obj_names:
                yield [getattr(model, name) for model in self.models]
            obj_list_names = self.obj_list_names[0]
            for name in obj_list_names:
                obj_lists = [getattr(model, name) for model in self.models]
                for i in range(obj_list_names[name]):
                    yield [obj_list[i] for obj_list in obj_lists]
            obj_dict_names = self.obj_dict_names[0]
            for name in obj_dict_names:
                obj_dicts = [getattr(model, name) for model in self.models]
                for key in obj_dict_names[name]:
                    yield [obj_dict[key] for obj_dict in obj_dicts]

        block_names = self.block_names[0]
        for name in block_names:
            for comp in ParallelComponentIterator(
                [getattr(model, name) for model in self.models],
                self.collect_vars,
                self.collect_constrs,
                self.collect_objs,
            ):
                yield comp
        block_list_names = self.block_list_names[0]
        for name in block_list_names:
            block_lists = [getattr(model, name) for model in self.models]
            for i in range(block_list_names[name]):
                for comp in ParallelComponentIterator(
                    [block_list[i] for block_list in block_lists],
                    self.collect_vars,
                    self.collect_constrs,
                    self.collect_objs,
                ):
                    yield comp
        block_dict_names = self.block_dict_names[0]
        for name in block_dict_names:
            block_dicts = [getattr(model, name) for model in self.models]
            for key in block_dict_names[name]:
                for comp in ParallelComponentIterator(
                    [block_dict[key] for block_dict in block_dicts],
                    self.collect_vars,
                    self.collect_constrs,
                    self.collect_objs,
                ):
                    yield comp

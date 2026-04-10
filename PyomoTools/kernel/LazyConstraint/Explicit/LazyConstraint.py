import pyomo.kernel as pmo


class lazy_constraint(pmo.constraint):
    """
    A lazy constraint is a constraint that is not added to the model until it is violated. This allows for a more efficient solution process, as the solver does not have to consider constraints that are not currently active.

    The main purpose of this class is to serve as a marker for a lazy constraint-enabled solver (like those found in this module) to identify which constraints should be treated as lazy constraints. The actual logic for checking if the constraint is violated and adding it to the model will be implemented in the solver's callback function.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

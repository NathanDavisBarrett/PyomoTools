"""
With this approach, constraints could be replaced with "lazy_constraint" objects and the GurobiPersistent_WithExplicitLazyConstraints should be used.

All further mechanics will be handled automatically using gurobi's natural handling of lazy constraints.
"""

from .Gurobi.GurobiExplicitLazy import GurobiPersistent_WithExplicitLazyConstraints
from .LazyConstraint import lazy_constraint

__all__ = ["GurobiPersistent_WithExplicitLazyConstraints", "lazy_constraint"]

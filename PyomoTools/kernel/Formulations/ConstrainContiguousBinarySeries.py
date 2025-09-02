import pyomo.kernel as pmo

from typing import Iterable

from .ConditionalGeq import ConditionalGeq
from .ConditionalLeq import ConditionalLeq


class ConstrainContiguousBinarySeries(pmo.block):
    """
    A block to model the following relationship in MILP form:

        Given X[t], a series of binary variables, only a contiguous sub-series of them can be 1-valued.

    Examples where this behavior arises include:
    * A series of binary variables representing whether or not a machine is on at each time step t. The machine can only turn on and off one time in the series.

    A very tight (though not necessarily compact) formulation of this relationship involves an Integer start and end index, StartIndex and EndIndex, such that X[i] = 1 if and only if StartIndex <= i <= EndIndex.
    """

    def __init__(
        self,
        X: Iterable[pmo.variable],
        minimumLength: int = 1,
        maximumLength: int = None,
    ):
        """
        Parameters
        ----------
        X: Iterable[pmo.variable]
            An iterable of Pyomo binary variables representing the series of binary variables to constrain.
        minimumLength: int (optional, Default=1)
            The minimum length of the contiguous sub-series that can be 1-valued.
        maximumLength: int (optional, Default=None)
            The maximum length of the contiguous sub-series that can be 1-valued. If None, there is no maximum length constraint.
        """
        super().__init__()

        X = list(X)
        n = len(X)

        minIndex = 0
        maxIndex = n - 1

        self.StartIndex = pmo.variable(
            domain=pmo.Integers, lb=minIndex, ub=maxIndex, value=minIndex
        )
        self.EndIndex = pmo.variable(
            domain=pmo.Integers, lb=minIndex, ub=maxIndex, value=maxIndex
        )

        self.MinimumLengthConstraint = pmo.constraint(
            expr=self.StartIndex <= self.EndIndex - minimumLength + 1
        )
        if maximumLength is not None:
            self.MaximumLengthConstraint = pmo.constraint(
                expr=self.EndIndex <= self.StartIndex + maximumLength - 1
            )

        self.IsAfterStart = pmo.variable_list(
            [pmo.variable(domain=pmo.Binary, value=0) for _ in range(n)]
        )
        self.IsBeforeEnd = pmo.variable_list(
            [pmo.variable(domain=pmo.Binary, value=0) for _ in range(n)]
        )

        self.AfterStartDefinition = pmo.block_list(
            [  # IsAfterStart[i] = 1 if StartIndex <= i
                ConditionalLeq(
                    A=self.StartIndex,
                    alpha=i,
                    X=self.IsAfterStart[i],
                    epsilon=1,  # 1 since the StartIndex is integer
                )
                for i in range(n)
            ]
        )
        self.BeforeEndDefinition = pmo.block_list(
            [  # IsBeforeEnd[i] = 1 if EndIndex >= i
                ConditionalGeq(
                    A=self.EndIndex,
                    alpha=i,
                    X=self.IsBeforeEnd[i],
                    epsilon=1,  # 1 since the EndIndex is integer
                )
                for i in range(n)
            ]
        )

        # X[i] is 1 if it is both after the start and before the end
        self.XDefinition = pmo.constraint_list(
            [
                pmo.constraint(X[i] == self.IsAfterStart[i] + self.IsBeforeEnd[i] - 1)
                for i in range(n)
            ]
        )

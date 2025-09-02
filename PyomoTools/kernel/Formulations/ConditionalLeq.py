from .ConditionalGeq import ConditionalGeq


class ConditionalLeq(ConditionalGeq):
    """
    A block to model the following relationship in MILP form:

        A > alpha  if X == 0
        A <= alpha if X == 1

    Or, more precisely:

        A >= alpha + epsilon  if X == 0
        A <= alpha            if X == 1

    where
    * A is a variable (real or integer)
    * X is a binary
    * alpha is a constant parameter
    * epsilon is a small positive constant


    Ideologically, this relationship is the same as ConditionalGeq, but with A' = 2*alpha - A
    """

    def TransformA(self, A, alpha):
        return 2 * alpha - A

    def __init__(self, A, alpha, X=None, epsilon=1e-5, A_bounds=None):
        Amin, Amax = self.GetBounds(A, A_bounds)

        Aprime = self.TransformA(A, alpha)
        AminPrime = self.TransformA(Amin, alpha)
        AmaxPrime = self.TransformA(Amax, alpha)

        super().__init__(Aprime, alpha, X, epsilon, (AmaxPrime, AminPrime))

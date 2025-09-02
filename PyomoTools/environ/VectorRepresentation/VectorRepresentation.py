import pyomo.environ as pyo
from pyomo.core.expr.numeric_expr import (
    MonomialTermExpression,
    NegationExpression,
    ProductExpression,
    SumExpression,
)
from pyomo.core.expr.numvalue import NumericConstant

from scipy.sparse import csr_matrix
import numpy as np


class LinkedListNode:
    def __init__(self, val, next=None, prev=None):
        self.val = val
        self.next = next
        self.prev = prev


class LinkedList:
    def __init__(self, lst):
        self.size = len(lst)
        self.headNode = None
        self.tailNode = None

    def append(self, val):
        newNode = LinkedListNode(val, prev=self.tailNode)
        if self.tailNode is not None:
            self.tailNode.next = newNode

        self.tailNode = newNode
        self.size += 1

        if self.headNode is None:
            self.headNode = self.tailNode

    def delete(self, node):
        prevNode = node.prev
        nextNode = node.next
        if prevNode is not None:
            prevNode.next = nextNode
        if nextNode is not None:
            nextNode.prev = prevNode

    def __len__(self):
        return self.size

    def __iter__(self):
        thisNode = self.headNode
        while thisNode is not None:
            yield thisNode
            thisNode = thisNode.next

    def iterval(self):
        thisNode = self.headNode
        while thisNode is not None:
            yield thisNode.val
            thisNode = thisNode.next


class VectorRepresentation:
    """
    A class to facilitate the vector representation of a pyomo model.

    Primary Components
    ------------------
    VAR_VEC: list
        A vector (python list) of all variables present in the model.
    CONSTR_VEC: list
        A vector (python list) of all constraints present in the model.

    VAR_VEC_INV: dict
        A dict mapping the string representation (i.e. name) of each variable to its index within VAR_VEC
    CONSTR_VEC_INV: dict
        A dict mapping the string representation (i.e. name) of each constraint to its index within CONSTR_VEC
    """

    def __init__(self, model: pyo.ConcreteModel):
        self.model = model

        # Convert variable bound and domain restrictions to explicit constraints.
        for var in self.model.component_objects(pyo.Var):
            if "Indexed" in str(type(var)):
                for idx in var:
                    if hasattr(idx, "__iter__"):
                        idx_str = f"{var}_{'_'.join(idx)}"
                    else:
                        idx_str = f"{var}_{idx}"
                    lb, ub = var[idx].bounds
                    if lb is not None:
                        setattr(
                            model, f"{idx_str}_LB", pyo.Constraint(expr=var[idx] >= lb)
                        )
                    if ub is not None:
                        setattr(
                            model, f"{idx_str}_UB", pyo.Constraint(expr=var[idx] <= ub)
                        )
            else:
                lb, ub = var.bounds
                if lb is not None:
                    setattr(model, f"{var}_LB", pyo.Constraint(expr=var >= lb))
                if ub is not None:
                    setattr(model, f"{var}_UB", pyo.Constraint(expr=var <= ub))

        self._Construct_VAR_VEC()
        self._Construct_CONSTR_VEC()

    def _Construct_VAR_VEC(self):
        numVar = 0
        for var in self.model.component_objects(pyo.Var):
            if "Indexed" in str(type(var)):
                for _ in var:
                    numVar += 1
            else:
                numVar += 1

        self.VAR_VEC = [None for _ in range(numVar)]
        self.VAR_VEC_INV = {}

        i = 0
        for var in self.model.component_objects(pyo.Var):
            if "Indexed" in str(type(var)):
                for idx in var:
                    vari = var[idx]
                    self.VAR_VEC[i] = vari
                    self.VAR_VEC_INV[str(vari)] = i
                    i += 1
            else:
                self.VAR_VEC[i] = var
                self.VAR_VEC_INV[str(var)] = i
                i += 1

    def _Construct_CONSTR_VEC(self):
        numConstr = 0
        for constr in self.model.component_objects(pyo.Constraint):
            if "Indexed" in str(type(constr)):
                for _ in constr:
                    numConstr += 1
            else:
                numConstr += 1

        self.CONSTR_VEC = [None for _ in range(numConstr)]
        self.CONSTR_VEC_INV = {}

        i = 0
        for constr in self.model.component_objects(pyo.Constraint):
            if "Indexed" in str(type(constr)):
                for idx in constr:
                    constri = constr[idx]
                    self.CONSTR_VEC[i] = constri
                    self.CONSTR_VEC_INV[str(constri)] = i
                    i += 1
            else:
                self.CONSTR_VEC[i] = constr
                self.CONSTR_VEC_INV[str(constr)] = i
                i += 1

    def Generate_Matrix_Representation(self):
        """
        A function to generate the matrix representation of a (mixed-integer) linear model:

        min c^T x + d
        s.t. S_leq A x <= S_leq b
             S_eq  A x == S_eq  b

        Here, S_leq and S_eq are row selection matrices that select the rows of A and b that correspond with "<=" and "==" constraints, respectively.

        Returns
        -------
        A,b,c,d: numpy arrays (b,c) or scipy csr matrix (A)
            The numpy arrays for the matrix representation of this problem. Please see the docstring for this function.
        inequalityIndices: constraint indices corresponding to inequality constraints
        equalityIndices: constraint indices corresponding to equality constraints
        """
        numConstr = len(self.CONSTR_VEC)
        numVar = len(self.VAR_VEC)
        A = LinkedList([])  # A linked list of [constrIndex,varIndex,value] lists

        b = np.empty(numConstr, dtype=float)
        c = np.zeros(numVar, dtype=float)
        equalityConstr = np.empty(numConstr, dtype=bool)

        for i, constr in enumerate(self.CONSTR_VEC):
            entries, const, equality = self._ParseConstraint(constr)
            for nodei in entries:
                A.append([i, *(nodei.val)])
            b[i] = -const
            equalityConstr[i] = equality

        # Now convert this A linked to a scipy sparse matrix
        rows = np.empty(len(A), dtype=int)
        cols = np.empty(len(A), dtype=int)
        data = np.empty(len(A), dtype=float)
        for i, (row, col, coef) in enumerate(A.iterval()):
            rows[i] = row
            cols[i] = col
            data[i] = coef
        A = csr_matrix((data, (rows, cols)), shape=(numConstr, numVar))

        numObj = 0
        for obj in self.model.component_objects(pyo.Objective, active=True):
            numObj += 1

        if numObj == 0:
            # No objective found
            # Do nothing
            d = 0
        elif numObj == 1:
            for obj in self.model.component_objects(pyo.Objective, active=True):
                entries, d = self._ParseExpression(obj.expr)
                for varIndex, coef in entries.iterval():
                    c[varIndex] = coef

                if obj.sense == pyo.maximize:
                    c *= -1  # Always assume minimization.
        else:
            raise Exception(
                f"Currently, only one objective is supported. {numObj} were detected."
            )

        equalityIndices = np.where(equalityConstr)[0]

        inequalityIndices = np.where(np.logical_not(equalityConstr))[0]

        return A, b, c, d, inequalityIndices, equalityIndices

    def _AddEntries(
        self, entries: LinkedList, const: float, newEntries: LinkedList, newConst: float
    ):
        """
        A function to handle the addition of a term to the linked list/const representation of an expression.

        Parameters
        ----------
        entries: LinkedList
            The variable coefficient entries already present in the expression.
        const: float
            The constant term of the expression already present.
        newEntries: LinkedList
            The coefficients to add to the expression (handled in-place)
        newConst: float
            The constant term to add to the expression. (not in-place)

        Returns
        -------
        const: float
            The new (summed) constant term for this expression.
        """
        # Iterate over the new entries and add any coefficients for variables already seen in this expression.
        ignoreIndices = []
        for i, nodei in enumerate(newEntries):
            coli, vali = nodei.val
            for nodej in entries:
                colj = nodej.val[0]
                if coli == colj:
                    nodej.val[1] += vali
                    ignoreIndices.append(i)

        # Add coefficients for variables not already seen in this expression.
        for i, nodei in enumerate(newEntries):
            if i in ignoreIndices:
                continue
            entries.append(nodei.val)

        return const + newConst

    def _ParseExpression(self, expr):
        """
        A function to turn a pyomo expression object into a sparse matrix representation of that expression

        Parameters
        ----------
        expr: pyomo expression, term, variable, or constant
            The expression you'd like to parse

        Returns
        -------
        entries: LinkedList
            A linked list of [varIndex,value] lists for each term in this expression.
        const: float
            Any constant terms remaining from this expression.
        """
        entries = LinkedList([])
        const = 0

        if hasattr(expr, "is_variable_type") and expr.is_variable_type():
            index = self.VAR_VEC_INV[str(expr)]
            entries.append([index, 1])
        elif isinstance(expr, SumExpression):
            for term in expr.args:
                newEntries, newConst = self._ParseExpression(term)
                const = self._AddEntries(entries, const, newEntries, newConst)

        elif isinstance(expr, int) or isinstance(expr, float):
            const += expr
        elif isinstance(expr, MonomialTermExpression):
            coef, var = expr.args
            index = self.VAR_VEC_INV[str(var)]
            entries.append([index, coef])
        elif isinstance(expr, NegationExpression):
            assert len(expr.args) == 1
            newEntries, newConst = self._ParseExpression(expr.args[0])

            newConst *= -1
            for nodei in newEntries:
                nodei.val[1] *= -1

            entries = newEntries
            const = newConst
        elif isinstance(expr, ProductExpression):
            results = [self._ParseExpression(term) for term in expr.args]

            # For this to be linear, no more than one of these results can have a non-zero number of entries. Track down which one this is and multiply it by the other coefficients.

            baseResult = None
            baseConst = 1

            newCoef = 1
            for newEntries, newConst in results:
                numNewEntries = len(newEntries)
                if numNewEntries > 0:
                    if baseResult is not None:
                        raise Exception(
                            "Bilinear term detected! Currently, VectorRepresentation can only handle linear models."
                        )
                    baseResult = newEntries
                    baseConst = newConst
                else:
                    newCoef *= newConst

            baseConst *= newCoef
            if baseResult is not None:
                for nodei in baseResult:
                    nodei.val[1] *= newCoef

            entries = baseResult
            const = baseConst

        else:
            raise Exception(
                f'Enable to parse expression "{expr}" of type "{type(expr)}". Is it Linear?'
            )

        return entries, const

    def _ParseConstraint(self, constr: pyo.Constraint):
        """
        A function to to transform a linear pyomo constraint into a

            A x <= b

        form.

        Here, A is a linked list of variable indices within the overall VAR_VEC paired with their coefficients and b is a constant.

        Parameters
        ----------
        constr: pyo.Constraint
            The constraint you'd like to parse

        Returns
        -------
        A: LinkedList
            A linked list of (index,coef) pairs for the coefficients of each variable in this constraint.
        b: float
            The constant term for this expression.
        relation: Bool
            True if this constraint is an equality contraint, False if it is a "<=" inequality.
        """
        lhs = constr.body
        # For now rhs will be zero

        upper = constr.upper
        lower = constr.lower
        relation = None

        if isinstance(upper, NumericConstant) or isinstance(upper, int):
            upper = float(upper)
        if isinstance(lower, NumericConstant) or isinstance(lower, int):
            lower = float(lower)

        if upper is None:
            if lower is None:
                raise Exception(
                    f"This constraint has no upper or lower bound!\n{constr}"
                )
            else:
                lhs -= lower
                lhs *= -1

                relation = False

        else:
            if lower is not None:
                assert np.allclose(
                    [
                        upper,
                    ],
                    [
                        lower,
                    ],
                ), f'Error! This constraint has an upper and lower bound that do not match. In essence, this is two constraints in one. This behavior is not supported at this time.\nConstraint: "{constr}"'
                lhs -= upper
                relation = True
            else:
                lhs -= upper
                relation = False

        # We now have lhs <== 0
        entires, const = self._ParseExpression(lhs)
        return entires, const, relation

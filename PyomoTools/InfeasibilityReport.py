import pyomo.environ as pyo
import re

from .GenerateExpressionString import GenerateExpressionStrings

class InfeasibilityReport:
    """
    A class to hold information pertaining to a set of violated constraints within a pyomo model.

    **NOTE #1**: Typically if a solver determines that a model is infeasible, it will not load a solution into the model object. Thus you cannot use this class in this case. However, if you manually load a solution into a model (e.g. by using the IO.LoadModelSolutionFromExcel function of this package), you and detect and infeasibilities that are present with that solution within this model

    **NOTE #2**: The generated report (from str(report) or from report.WriteFile) are structured as follows: Each violated constraint has 4 lines:
        1) The fully symbolic expression
        2) The symbolic expression with all values substituted in but with whitespace conserved for ease of comparison with the fully symbolic expression
        3) The substituted expression but with all unnecessary white space removed
        4) The substituted expression evaluated down to it's right-hand-side and left-hand-side

    Constructor Parameters
    ----------------------
    model: pyo.ConcreteModel
        The pyomo model (containing a solution) that you'd like to generate the infeasibility report for.
    aTol: float (optional, Default = 1e-3)
        The absolute tolerance to use when evaluating whether or not a given constraint is violated or not.
    onlyInfeasibilities: bool (optional, Default = True)
        An indication that you'd only like infeasibilities listed in this report. If False, all constraints will in included in the report.

    Members:
    --------
    exprs: 
        A dict mapping the name of each violated constraint (str) to either that constraint's expression string or another dict that maps that constraint's indices to those indices' expression strings
    substitutedExprs: 
        A dict with the same structure as exprs but with the value of each variable substituted into the expression string.
    """
    def __init__(self, model:pyo.ConcreteModel,aTol=1e-3,onlyInfeasibilities=True):
        self.exprs = {}
        self.substitutedExprs = {}
        self.onlyInfeasibilities = onlyInfeasibilities

        self.numInfeas = 0

        for c in model.component_objects(pyo.Constraint, active=True):
            try:
                constr = getattr(model,str(c))
            except:
                if ".DCC_constraint" in str(c):
                    continue
                print(f"Warning! Could not locate constraint named \"{c}\"")
                continue

            if "Indexed" in str(type(constr)):
                #This constraint is indexed
                for index in constr:
                    if not self.TestFeasibility(constr[index],aTol=aTol):
                        self.AddInfeasibility(name=c,index=index,constr=constr[index])
            else:
                if not self.TestFeasibility(constr,aTol=aTol):
                    self.AddInfeasibility(name=c,constr=constr)

    def TestFeasibility(self,constr:pyo.Constraint,aTol=1e-5):
        """
        A function to test whether or not a given constraint is violated by the solution contained within it.

        Parameters
        ----------
        constr: pyo.Constraint
            The constraint object you'd like to test.
        aTol: float (optional, Default = 1e-3)
            The absolute tolerance to use when evaluating whether or not a given constraint is violated or not.

        Returns
        bool:
            True if the solution is feasible with respect to this constraint or False if it is not.
        """
        if not self.onlyInfeasibilities:
            return False

        lower = constr.lower
        upper = constr.upper
        value = pyo.value(constr)
        
        if lower is not None:
            if value < lower - aTol:
                return False
        if upper is not None:
            if value > upper + aTol:
                return False
        return True

    def AddInfeasibility(self,name:str,constr:pyo.Constraint,index:object=None):
        """
        A function to add a violated constraint to this report

        Parameters:
        -----------
        name: str
            The name of the violated constraint
        constr: pyo.Constraint
            The constraint object
        index: object (optional, Default=None)
            If the constraint is indexed, pass the appropriate index here.
        """
        self.numInfeas += 1
        if index is None:
            self.exprs[name], self.substitutedExprs[name] = GenerateExpressionStrings(constr.expr)
        else:
            if name not in self.exprs:
                self.exprs[name] = {}
                self.substitutedExprs[name] = {}
            self.exprs[name][index], self.substitutedExprs[name][index] = GenerateExpressionStrings(constr.expr)

    def Iterator(self):
        """
        A python generator object (iterator) that iterates over each infeasibility.

        Iterates are strings of the following format: ConstraintName[Index (if appropriate)]: Expr \n SubstitutedExpression
        """
        for c in self.exprs:
            if isinstance(self.exprs[c],dict):
                for i in self.exprs[c]:
                    varName = "{}[{}]:".format(c,i)

                    spaces = " "*len(varName)
                    shortenedStr = re.sub(' +', ' ', self.substitutedExprs[c][i])
                    dividers = ["==","<=",">="]
                    divider = None
                    for d in dividers:
                        if d in shortenedStr:
                            divider = d
                            break
                    
                    if divider is None:
                        raise Exception(f"The following expression is not well posed as a constraint!\n{shortenedStr}")
                    
                    divIndex = shortenedStr.index(divider)
                    lhs = shortenedStr[:divIndex].lstrip()
                    rhs = shortenedStr[divIndex+2:].lstrip()
                    lhsVal = eval(lhs)
                    rhsVal = eval(rhs)

                    evalStr = f"{lhsVal} {divider} {rhsVal}"

                    yield "{} {}\n{} {}\n{} {}\n{} {}".format(varName,self.exprs[c][i],spaces,self.substitutedExprs[c][i],spaces,shortenedStr,spaces,evalStr)
            else:
                spaces = " "*len(c)
                shortenedStr = re.sub(' +', ' ', self.substitutedExprs[c])
                dividers = ["==","<=",">="]
                divider = None
                for d in dividers:
                    if d in shortenedStr:
                        divider = d
                        break
                
                if divider is None:
                    raise Exception(f"The following expression is not well posed as a constraint!\n{shortenedStr}")
                
                divIndex = shortenedStr.index(divider)
                lhs = shortenedStr[:divIndex].lstrip()
                rhs = shortenedStr[divIndex+2:].lstrip()
                lhsVal = eval(lhs)
                rhsVal = eval(rhs)

                evalStr = f"{lhsVal} {divider} {rhsVal}"

                yield "{}: {}\n{}  {}\n{}  {}\n{}  {}".format(c,self.exprs[c],spaces,self.substitutedExprs[c],spaces,shortenedStr,spaces,evalStr)

    def __len__(self):
        return self.numInfeas
    
    def __str__(self):
        """
        A function to convert this report to a string.

        Usage
        -----
        result = str(reportObject)
        """
        lines = []
        for infeas in self.Iterator():
                lines.append(infeas)
                lines.append("\n\n")
        return '\n'.join(lines)

    def WriteFile(self,fileName:str):
        """
        A function to write the output to a file.
        """
        with open(fileName,"w") as f:
            f.write(str(self))
        
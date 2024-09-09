import pyomo.environ as pyo
from pyomo.core.expr.current import identify_variables
from typing import Union

class Object:
    pass

def SubstituteExpression(expr:Union[pyo.Expression,pyo.Var], varMap:dict):
    """
    A function to execute a series of substitutions within a pyomo expression

    Parameters
    ----------
    expr: pyo.Expression or pyo.Variable
        The expression you'd like to substitute. Note that some expressions might present as simple variables. We handle this case.
    varMap: dict:
        A dict mapping variables from within the expr to the object they should be substituted with.
    """
    if expr.is_expression_type():
        return expr.clone(substitute=varMap)
    elif expr.is_variable_type():
        return varMap[expr]
    else:
        raise Exception("Unable to recognize the inputted expr.")

def MergeModels(outerModel:pyo.ConcreteModel,innerModels:dict):
    """
    A function to merge a collection of models into one model.

    Sub-models will be Namespaced via two options, both will be accessible:
        1. outerModel.innerModel.attribute
        2. outerModel.innerModel_attribute

    Each attribute of the inner models will be duplicated into the inner model. In other words, an entire new copy of each inner Model will be created. The "innerModel" attribute within the outerModel will simply be a a python "object" class instance. Each attribute of this "Object" object (e.g. innerModel.MyVariable) will simply point to the copy of that attribute belonging to the outerModel under the name "outerModel.innerModel_MyVariable.

    Parameters
    ----------
    outerModel: pyo.ConcreteModel
        The model within which you'd like to add each innerModel
    innerModels: dict
        A dict mapping the name of each inner model (str) to that particular innerModel (pyo.ConcreteModel). The keys of this dict will be the names by which you can reference members of that particular inner model. For example if I pass {"model1": ___}, I could reference the attributes of this model via outerModel.model1.___.

    Returns
    -------
    None (This function simply edits the existing outerModel object)
    """
    for innerModelName in innerModels:
        innerModel = innerModels[innerModelName]

        innerModelObject = Object()

        #First copy over all sets
        for s in innerModel.component_objects(pyo.Set):
            elements = [e for e in s]
            name = str(s)
            newName = f"{innerModelName}_{name}"

            setattr(outerModel,newName,pyo.Set(initialize=elements))
            setattr(innerModelObject,name,getattr(outerModel,newName))

        #Next, copy over all variables
        varMap = {}
        for var in innerModel.component_objects(pyo.Var):
            name = str(var)
            newName = f"{innerModelName}_{name}"

            isIndexed = "Indexed" in str(type(var))

            if not isIndexed:
                setattr(outerModel,newName,pyo.Var(domain=var.domain,initialize=var.value,bounds=var.bounds))
                newVar = getattr(outerModel,newName)
                if not var.active:
                    newVar.deactivate()
                setattr(innerModelObject,name,newVar)
                varMap[name] = newVar
            else:
                indices = var.index_set()
                varSample = var[indices[1]]
                setattr(outerModel,newName,pyo.Var(indices,domain=varSample.domain,bounds=varSample.bounds))
                newVar = getattr(outerModel,newName)
                for i in indices:
                    newVar[i].value = var[i].value
                    varMap[str(var[i])] = newVar[i]
                if not var.active:
                    newVar.deactivate()
                setattr(innerModelObject,name,newVar)
                varMap[name] = newVar

        #Next, copy over all constraints
        for constr in innerModel.component_objects(pyo.Constraint):
            name = str(constr)
            newName = f"{innerModelName}_{name}"

            isIndexed = "Indexed" in str(type(constr))

            if not isIndexed:
                expr = constr.expr
                varsInExpr = list(identify_variables(expr))
                subVarMap = {id(var): varMap[str(var)] for var in varsInExpr}

                newExpr = SubstituteExpression(expr,subVarMap)
                setattr(outerModel,newName,pyo.Constraint(expr=newExpr))
                newConstr = getattr(outerModel,newName)
                if not constr.active:
                    newConstr.deactivate()

                setattr(innerModelObject,name,newConstr)
            else:
                indices = constr.index_set()
                
                def rule(outerModel,*index):
                    expr = constr[index].expr
                    varsInExpr = list(identify_variables(expr))
                    subVarMap = {id(var): varMap[str(var)] for var in varsInExpr}

                    newExpr = SubstituteExpression(expr,subVarMap)
                    return newExpr
                setattr(outerModel,newName,pyo.Constraint(indices,rule=rule))
                newConstr = getattr(outerModel,newName)
                if not constr.active:
                    newConstr.deactivate()

                setattr(innerModelObject,name,newConstr)

        #NOTE: Objectives from each innerModel will be ignored.

        #Finally, place the innerModelObject into the outerModel
        setattr(outerModel,innerModelName,innerModelObject)
import pyomo.environ as pyo
from pyomo.core.expr.current import identify_variables
from typing import Union
from warnings import warn

class MergableModel(pyo.ConcreteModel):
    """
    A extension of the pyomo ConcreteModel class that allows sub-models to be added.

    Sub-models will be Namespaced via two options, both will be accessible:
        1. parentModel.subModel.attribute
        2. parentModel._SUBMsubModel_COMPattribute

    Each attribute of the sub-model models will be duplicated into this parent model. In other words, an entire new copy of each of the attributes of each sub-model will be created. The "subModel" attribute within the parent will simply be a a python "object" class instance. Each attribute of this "Object" object (e.g. subModel.MyVariable) will simply point to the copy of that attribute belonging to the parent model under the name "parentModel.subModel_MyVariable.    
    """

    subModelKeyword = "_SUBMODEL_"
    componentKeyword = "_COMPONENT_"

    def __init__(self):
        super().__init__()

    class Object:
        pass

    def _SubstituteExpression(self,expr:Union[pyo.Expression,pyo.Var], varMap:dict):
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

    def _Namespace(self,subModelName:str,attrName:str):
        """
        A function to define a custom attribute name for an attribute of a sub model.

        This is important since, in order to preserve the parentModel.subModel.attribute architecture across multiple levels of sub-models, we will simply look for any attribute containing the keywords _SUBM and _COMP
        """
        return f"{self.subModelKeyword}{subModelName}{self.componentKeyword}{attrName}"

    def AddSubModel(self,subModelName:str,subModel:pyo.ConcreteModel):
        """
        A function to add a sub-model to this model.

        Remember that the sub-model's representation in this parent model will be an entirely new copy. Any changes made to the "subModel" object provided will not be reflected in the parent model's copy of the sub-model.

        Parameters
        ----------
        subModelName: str
            The name by which this sub-model will be addressed within the scope of the parent model. (Cannot contain non-alphanumeric characters and must begin with an alphabetic character)
        subModel: pyo.ConcreteModel or MergableModel
            The sub-model to be added to this parent model.
        """
        #First copy over all sets
        for s in subModel.component_objects(pyo.Set):
            elements = [e for e in s]
            name = str(s)
            newName = self._Namespace(subModelName,name)

            setattr(self,newName,pyo.Set(initialize=elements))

        #Next, copy over all variables
        varMap = {}
        for var in subModel.component_objects(pyo.Var):
            name = str(var)
            newName = self._Namespace(subModelName,name)

            isIndexed = "Indexed" in str(type(var))

            if not isIndexed:
                setattr(self,newName,pyo.Var(domain=var.domain,initialize=var.value,bounds=var.bounds))
                newVar = getattr(self,newName)
                if not var.active:
                    newVar.deactivate()
                varMap[name] = newVar
            else:
                indices = var.index_set()
                if len(indices) == 0:
                    setattr(self,newName,pyo.Var(indices))
                else:
                    varSample = var[indices[1]]
                    setattr(self,newName,pyo.Var(indices,domain=varSample.domain,bounds=varSample.bounds))
                newVar = getattr(self,newName)
                for i in indices:
                    newVar[i].value = var[i].value
                    varMap[str(var[i])] = newVar[i]
                if not var.active:
                    newVar.deactivate()
                varMap[name] = newVar

        #Next, copy over all constraints
        for constr in subModel.component_objects(pyo.Constraint):
            name = str(constr)
            newName = self._Namespace(subModelName,name)

            isIndexed = "Indexed" in str(type(constr))

            if not isIndexed:
                expr = constr.expr
                varsInExpr = list(identify_variables(expr))
                subVarMap = {id(var): varMap[str(var)] for var in varsInExpr}

                newExpr = self._SubstituteExpression(expr,subVarMap)
                setattr(self,newName,pyo.Constraint(expr=newExpr))
                newConstr = getattr(self,newName)
                if not constr.active:
                    newConstr.deactivate()

            else:
                indices = constr.index_set()
                
                def rule(self,*index):
                    if len(index) == 1:
                        index = index[0]
                    if index not in constr:
                        return pyo.Constraint.Feasible
                    expr = constr[index].expr
                    varsInExpr = list(identify_variables(expr))
                    subVarMap = {id(var): varMap[str(var)] for var in varsInExpr}

                    newExpr = self._SubstituteExpression(expr,subVarMap)
                    return newExpr
                setattr(self,newName,pyo.Constraint(indices,rule=rule))
                newConstr = getattr(self,newName)
                if not constr.active:
                    newConstr.deactivate()


        #NOTE: Objectives from each subModel will be ignored.

        self.UpdateSubModelObjects(subModelName)

    def UpdateSubModelObjects(self,targetSubModel:str=None):
        """
        In order to handle multiple layers of sub-models, we'll simply search for the keywords "_SUBM" and "_COMP" in each of this model's attributes. If the keyword appears, we'll create a reference to that attribute in this model's attributes.

        For example, if we have three levels of sub models ending in a variable called "X", I'd expect something like the following.

            self._SUBMsubModel1_COMP_SUBMsubmodel2_COMP_SUBMsubmodel3_COMPX

        The copy of this variable should be referenced as follows.

            self.subModel1.subModel2.subModel3.X

        Parameters
        ----------
        targetSubModel: str (optional, Default = None)
            If you'll be adding multiple sub-models at different times, this function can potentially waste time, re-assigning existing sub-model references. Thus, if you'd like to restrict the scope of this function to attributes belonging to only one sub-model (an immediate child of this parent model), you can indicate the name of that sub-model here. If None is provided, the update will be executed for all sub-model attribute references.
        """
        if targetSubModel is not None:
            targetSubModel = f"{self.subModelKeyword}{targetSubModel}"

        for typ in [pyo.Set,pyo.Var,pyo.Constraint]:
            for attr in self.component_objects(typ):
                attrName = str(attr)

                if targetSubModel is not None:
                    if not attrName.startswith(targetSubModel):
                        continue

                if self.subModelKeyword in attrName and self.componentKeyword in attrName:
                    subModelPath = []
                    terminalObjName = None

                    strIndex = 0
                    while True:
                        #Step1: Skip the subModelKeyword
                        strIndex += len(self.subModelKeyword)

                        #Step2: Detect where the next componentKeyword is
                        endIndex = attrName.find(self.componentKeyword, strIndex)

                        if endIndex == -1:
                            raise Exception(f"SubModel parent attribute is not correctly formatted: Opening subModelKeyword but no closing componentKeyword! \"{attrName}\"")
                        
                        #Step 3: Select all the text in between, this is the subModel name
                        subModelName = attrName[strIndex:endIndex]
                        subModelPath.append(subModelName)

                        #Step 4: Skip the following componentKeyword
                        strIndex = endIndex + len(self.componentKeyword)

                        #Step 5: Detect the next subModelKeyword is
                        endIndex = attrName.find(self.subModelKeyword,strIndex)

                        #Step 6: Select the text in between and deal with it accordingly
                        if endIndex == -1:
                            #This indicates that this is the terminalObjName
                            terminalObjName = attrName[strIndex:]
                            break
                        else:
                            #This indicates that this is the name of the next subModel. It will be collected on the next iteration.
                            continue


                    itrObj = self
                    for p in subModelPath:
                        if not hasattr(itrObj,p):
                            setattr(itrObj,p,self.Object())
                        itrObj = getattr(itrObj,p)

                    if not hasattr(itrObj,terminalObjName):
                        setattr(itrObj,terminalObjName,attr)

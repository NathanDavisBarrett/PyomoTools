import pyomo.environ as pyo
import pyomo.kernel as pmo
import numpy as np
import warnings

from typing import Union

def SendErrorWarning(str,error,warning):
    if error:
        raise AssertionError(str)
    if warning:
        warnings.warn(str)

def AssertPyomoModelsEqual_Kernel(model1:pmo.block,model2:pmo.block,rtol=1e-5,atol=1e-8,error=False,warning=False) -> bool:
    """
    A function to determine if two pyomo.kernel models contain equal solutions.

    An equal solution is defined as follows.
        A) Both models have the same variables with the same names
        B) Each variable's value within model1 is within an "rtol" relative tolerance OR an "atol" absolute tolerance to that same variable's value in model 2.
    
    Parameters
    ----------
    model1: pyo.block
        The 1st pyomo model you'd like to compare
    model2: pyo.block
        The 2nd pyomo model you'd like to compare
    rtol: float (optional, Default: 1e-5)
        The relative tolerance you'd like to use for each comparison. Recall that model1's values will be taken as the divisors.
    atol: float (optional, Default: 1e-8)
        The relative tolerance you'd like to use for each comparison. Recall that model1's values will be taken as the divisors.
    error: bool (optional, Default: False)
        A boolean indicating whether or not you'd like to throw an error if a mismatch is found.
    warning: bool (optional, Default: False)
        A boolean indicating whether or not you'd like to raise a warning if a mismatch is found.
    
    Returns
    -------
    bool:
        A boolean indicating whether or not the two models are equal
    """
    variableNames1 = set([])
    subBlockNames1 = set([])
    for v1 in model1.children():
        if isinstance(v1,(pmo.variable,pmo.variable_list,pmo.variable_tuple,pmo.variable_dict)):
            variableNames1.add(v1.getname())
        if isinstance(v1,(pmo.block,pmo.block_list,pmo.block_tuple,pmo.block_dict)):
            subBlockNames1.add(v1.getname())
    
    variableNames2 = set([])
    subBlockNames2 = set([])
    for v2 in model2.children():
        if isinstance(v2,(pmo.variable,pmo.variable_list,pmo.variable_tuple,pmo.variable_dict)):
            variableNames2.add(v2.getname())
        if isinstance(v2,(pmo.block,pmo.block_list,pmo.block_tuple,pmo.block_dict)):
            subBlockNames2.add(v2.getname())

    if variableNames1 != variableNames2:
        if error or warning:
            variablesIn1ButNot2 = variableNames1 - variableNames2
            variablesIn2ButNot1 = variableNames2 - variableNames1
            SendErrorWarning("The following variables are present in model1 but not model2:\n{}\n\nThe following variables are present in model2 but not model1:\n{}".format(variablesIn1ButNot2,variablesIn2ButNot1),error,warning)
        return False
    
    if subBlockNames1 != subBlockNames2:
        if error or warning:
            subBlocksIn1ButNot2 = subBlockNames1 - subBlockNames2
            subBlocksIn2ButNot1 = subBlockNames2 - subBlockNames1
            SendErrorWarning("The following sub-blocks are present in model1 but not model2:\n{}\n\nThe following sub-blocks are present in model2 but not model1:\n{}".format(subBlocksIn1ButNot2,subBlocksIn2ButNot1),error,warning)
        return False
    
    
    for v in variableNames1:
        v1 = getattr(model1,v)
        v2 = getattr(model2,v)

        if type(v1) != type(v2):
            if error or warning:
                SendErrorWarning(f"Variable {v} has a different type in model1 ({type(v1)}) and model2 ({type(v2)}).",error,warning)
            return False

        if isinstance(v1,pmo.variable):
            val1 = v1.value
            val2 = v2.value
            if val1 is not None and val2 is not None:
                if not np.allclose([val1,],[val2,],rtol=rtol,atol=atol):
                    if error or warning:
                        SendErrorWarning(f"The values for {v} do not match! model1: {val1}, model2: {val2}",error,warning)
                    return False
            elif val1 is not None or val2 is not None:
                if error or warning:
                    SendErrorWarning(f"The values for {v} do not match! model1: {val1}, model2: {val2}",error,warning)
                return False
        elif isinstance(v1,(pmo.variable_list,pmo.variable_tuple)):
            if len(v1) != len(v2):
                if error or warning:
                    SendErrorWarning(f"Variable {v} has a different length in model1 ({len(v1)}) and model2 ({len(v2)}).",error,warning)
                return False
            for i in range(len(v1)):
                val1 = v1[i].value
                val2 = v2[i].value
                if val1 is not None and val2 is not None:
                    if not np.allclose([val1,],[val2,],rtol=rtol,atol=atol):
                        if error or warning:
                            SendErrorWarning(f"The values for {v}[{i}] do not match! model1: {val1}, model2: {val2}",error,warning)
                        return False
                elif val1 is not None or val2 is not None:
                    if error or warning:
                        SendErrorWarning(f"The values for {v}[{i}] do not match! model1: {val1}, model2: {val2}",error,warning)
                    return False
        elif isinstance(v1,pmo.variable_dict):
            if v1.keys() != v2.keys():
                if error or warning:
                    SendErrorWarning(f"Variable {v} has different keys in model1 ({set(v1.keys())}) and model2 ({set(v2.keys())}).",error,warning)
                return False
            for i in v1.keys():
                val1 = v1[i].value
                val2 = v2[i].value
                if val1 is not None and val2 is not None:
                    if not np.allclose([val1,],[val2,],rtol=rtol,atol=atol):
                        if error or warning:
                            SendErrorWarning(f"The values for {v}[{i}] do not match! model1: {val1}, model2: {val2}",error,warning)
                        return False
                elif val1 is not None or val2 is not None:
                    if error or warning:
                        SendErrorWarning(f"The values for {v}[{i}] do not match! model1: {val1}, model2: {val2}",error,warning)
                    return False
        else:
            raise TypeError(f"Unable to handle variable type {type(v1)} for variable {v}")

    for subBlock in subBlockNames1:
        print("NOW TESTING:",subBlock)
        subBlock1 = getattr(model1,subBlock)
        subBlock2 = getattr(model2,subBlock)

        if type(subBlock1) != type(subBlock2):
            if error or warning:
                SendErrorWarning("Sub-block {} has a different type in model1 ({type(subBlock1)}) and model2 ({type(subBlock2)}).".format(subBlock),error,warning)
            return False

        if isinstance(subBlock1,pmo.block):
            result = AssertPyomoModelsEqual_Kernel(subBlock1,subBlock2)
            if not result:
                return False
        elif isinstance(subBlock1,(pmo.block_list,pmo.block_tuple)):
            if len(subBlock1) != len(subBlock2):
                if error or warning:
                    SendErrorWarning(f"Sub-block {subBlock} has a different length in model1 ({len(subBlock1)}) and model2 ({len(subBlock2)}).",error,warning)
                return False
            for i in range(len(subBlock1)):
                result = AssertPyomoModelsEqual_Kernel(subBlock1[i],subBlock2[i])
                if not result:
                    return False
        elif isinstance(subBlock1,pmo.block_dict):
            if subBlock1.keys() != subBlock2.keys():
                if error or warning:
                    SendErrorWarning(f"Sub-block {subBlock} has different keys in model1 ({set(subBlock1.keys())}) and model2 ({set(subBlock2.keys())}).",error,warning)
            for i in subBlock1.keys():
                result = AssertPyomoModelsEqual_Kernel(subBlock1[i],subBlock2[i])
                if not result:
                    return False
                

    return True 

def AssertPyomoModelsEqual_Environ(model1:pyo.ConcreteModel,model2:pyo.ConcreteModel,rtol=1e-5,atol=1e-8,error=False,warning=False) -> bool:
    """
    A function to determine if two pyomo.environ models contain equal solutions.

    An equal solution is defined as follows.
        A) Both models have the same variables with the same names
        B) Each variable's value within model1 is within an "rtol" relative tolerance OR an "atol" absolute tolerance to that same variable's value in model 2.
    
    Parameters
    ----------
    model1: pyo.ConcreteModel
        The 1st pyomo model you'd like to compare
    model2: pyo.ConcreteModel
        The 2nd pyomo model you'd like to compare
    rtol: float (optional, Default: 1e-5)
        The relative tolerance you'd like to use for each comparison. Recall that model1's values will be taken as the divisors.
    atol: float (optional, Default: 1e-8)
        The relative tolerance you'd like to use for each comparison. Recall that model1's values will be taken as the divisors.
    error: bool (optional, Default: False)
        A boolean indicating whether or not you'd like to throw an error if a mismatch is found.
    warning: bool (optional, Default: False)
        A boolean indicating whether or not you'd like to raise a warning if a mismatch is found.
    
    Returns
    -------
    bool:
        A boolean indicating whether or not the two models are equal
    """
    variableNames1 = set([])
    for v1 in model1.component_objects():
        if v1.type() is pyo.Var:
            variableNames1.add(v1.getname())
    
    variableNames2 = set([])
    for v2 in model2.component_objects():
        if v2.type() is pyo.Var:
            variableNames2.add(v2.getname())

    if variableNames1 != variableNames2:
        if error or warning:
            variablesIn1ButNot2 = variableNames1 - variableNames2
            variablesIn2ButNot1 = variableNames2 - variableNames1
            SendErrorWarning("The following variables are present in model1 but not model2:\n{}\n\nThe following variables are present in model2 but not model1:\n{}".format(variablesIn1ButNot2,variablesIn2ButNot1),error,warning)
        return False
    
    for v in variableNames1:
        v1 = model1.find_component(v)
        v2 = model2.find_component(v)

        v1Indices = set([i for i in v1])
        v2Indices = set([i for i in v2])

        if v1Indices != v2Indices:
            if error or warning:
                SendErrorWarning(f"The indices for variable \"{v}\" are not consistent between the two models.\nModel 1 Indices: {v1Indices}\nModel 2 Indices: {v2Indices}",error,warning)
            return False


        for index in v1:
            val1 = v1[index].value
            val2 = v2[index].value

            if val1 is None or val2 is None:
                if val1 is not None or val2 is not None:
                    if error or warning:
                        SendErrorWarning("The values for {}[{}] do not match! model1: {}, model2: {}".format(v,index,val1,val2),error,warning)
                    return False
                continue
                    

            if val1 == 0:
                if val2 != 0:
                    if np.abs(val2) < atol:
                        continue
                    if error or warning:
                        SendErrorWarning("The values for {}[{}] do not match! model1: {}, model2: {}".format(v,index,val1,val2),error,warning)
                    return False
            else:
                aerr = np.abs(val1 - val2)
                rerr = np.abs((val1 - val2)/val1)
                if not (rerr < rtol or aerr < atol):
                    if error or warning:
                        SendErrorWarning("The values for {}[{}] do not match! model1: {}, model2: {}".format(v,index,val1,val2),error,warning)
                    return False
    
    return True

def AssertPyomoModelsEqual(model1:Union[pmo.block,pyo.ConcreteModel],model2:Union[pmo.block,pyo.ConcreteModel],rtol=1e-5,atol=1e-8,error=False,warning=False) -> bool:
    """
    A function to determine if two pyomo.kernel or pyomo.environ models contain equal solutions.

    An equal solution is defined as follows.
        A) Both models have the same variables with the same names
        B) Each variable's value within model1 is within an "rtol" relative tolerance OR an "atol" absolute tolerance to that same variable's value in model 2.
    
    Parameters
    ----------
    model1: pyo.block | pyo.ConcreteModel
        The 1st pyomo model you'd like to compare
    model2: pyo.block | pyo.ConcreteModel
        The 2nd pyomo model you'd like to compare
    rtol: float (optional, Default: 1e-5)
        The relative tolerance you'd like to use for each comparison. Recall that model1's values will be taken as the divisors.
    atol: float (optional, Default: 1e-8)
        The relative tolerance you'd like to use for each comparison. Recall that model1's values will be taken as the divisors.
    error: bool (optional, Default: False)
        A boolean indicating whether or not you'd like to throw an error if a mismatch is found.
    warning: bool (optional, Default: False)
        A boolean indicating whether or not you'd like to raise a warning if a mismatch is found.
    
    Returns
    -------
    bool:
        A boolean indicating whether or not the two models are equal
    """
    assert type(model1) == type(model2), f"Models must be of the same type, not {type(model1)} and {type(model2)}"

    if isinstance(model1,pmo.block):
        return AssertPyomoModelsEqual_Kernel(model1,model2,rtol=rtol,atol=atol,error=error,warning=warning)
    elif isinstance(model1,pyo.ConcreteModel):
        return AssertPyomoModelsEqual_Environ(model1,model2,rtol=rtol,atol=atol,error=error,warning=warning)
    else:
        raise TypeError(f"Unable to handle models of type {type(model1)}")


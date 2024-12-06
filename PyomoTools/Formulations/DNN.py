import pyomo.environ as pyo
from enum import Enum
import numpy as np

# from ..LoadIndexedSet import LoadIndexedSet

class DNN_Activation(Enum):
    RelU = 1
    LeakyRelU = 2



class DNN:
    """
    A class that generates a deep neural network relating the input variables "x" to the output variables "y" in a given number of layers with given sizes. In between each layer, a given activation function will be applied element-wise.

    Parameters
    ----------
    model: pyo.ConcreteModel
        The model within which this DNN should live
    xVars: list
        A list of variables within the model that are the inputs to this DNN.
    yVars: list
        A list of output variables within the model that are the outputs of this DNN.
    layerSizes: list
        A list of ints. Each value represents the size of this layer (i.e. number of variables to include)
    relationshipName: str 
        The name of this relationship (which will be part of all the variables and constraints used to define this relationship.)
    activation: DNN_Activation (optional, Default=RelU)
        The nonlinear activation function to use in this model.
    """
    def __init__(self,model:pyo.ConcreteModel,xVars:list,yVars:list,layerSizes:list,relationshipName:str,activation:DNN_Activation=DNN_Activation.RelU):
        perceptronName = f"{relationshipName}_Perceptron"

        setattr(model,f"{relationshipName}_Layers",pyo.Set(initialize=list(range(len(layerSizes)))))

        allNodeIndices = []
        layerSets = {}
        for i in range(len(layerSizes)):
            layerSets[i] = list(range(layerSizes[i]))
            allNodeIndices.extend(layerSets[i])

        edges = {}
        
        #First add x to L1
        for iNode in layerSets[0]:
            ii = (0,iNode)
            edges[ii] = [(*ii,j) for j in range(len(xVars))]
        #Now add Li to Lj
        for l in range(1,len(layerSizes)):
            for iNode in layerSets[l]:
                ii = (l,iNode)
                edges[ii] = [(*ii,j) for j in layerSets[l-1]]
        #Now add L-1 to y
        for iNode in range(len(yVars)):
            ii = (len(layerSizes),iNode)
            edges[ii] = [(*ii,j) for j in layerSets[len(layerSizes)-1]]

        allEdges = []
        for ii in edges:
            allEdges.extend(edges[ii])
        
        weightName = f"{relationshipName}_W"
        setattr(model,weightName,pyo.Var(allEdges,domain=pyo.Reals))



def test():
    model = pyo.ConcreteModel()
    model.X = pyo.Var([0,1,2,3])
    model.Y = pyo.Var([0,1,2,3])

    dnn = DNN(model,[model.X[i] for i in [0,1,2,3]],[model.Y[i] for i in [0,1,2,3]],[10,20,10],"X-Y")

    

test()

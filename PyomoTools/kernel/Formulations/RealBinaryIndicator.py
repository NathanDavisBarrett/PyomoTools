import pyomo.kernel as pmo

import numpy as np
import enum

from typing import Union, Tuple
from warnings import warn

from ._Formulation import _Formulation

class ARelationOption(enum.Enum):
    GEQ = 0
    LEQ = 1

class RealBinaryIndicator(_Formulation):
    """
    A block to model the following relationship in MILP form:

        X = 1 if (A >= alphaPrime) and (Y = 1) else 0

    or 

        X = 1 if (A <= alphaPrime) and (Y = 1) else 0

    (depending of the ARelationOption provided)

    where
    * A is a continuous variable (or expression) between alphaMin and alphaMax
    * alphaMin <= alphaPrime <= alphaMax are constant parameters
    * Y is a binary variable (or expression that evaluates to binary)
    * X is a binary variable (or expression that evaluates to binary)

    This is accomplished by constraining three planes defined by the following collections of (X,Y,A) points:

    ARelationOption.GEQ:
        1. (0,0,alphaMin), (0,1,alphaMin), (1,1,alphaPrime)
        2. (0,0,alphaMin), (1,1,alphaMin), (0,0,alphaMax)
        3. (0,1,alphaPrime-epsilon), (0,0,alphaMax), (1,1,alphaMax)

    ARelationOption.LEQ:
        1. (0,0,alphaMin), (0,1,alphaPrime+epsilon), (1,1,alphaMin)
        2. (0,0,alphaMin), (1,1,alphaMin), (0,0,alphaMax)
        3. (0,1,alphaMax), (1,1,alphaPrime), (0,0,alphaMax)
    """
    def __init__(self,
            X:Union[pmo.variable, pmo.expression],
            Y:Union[pmo.variable, pmo.expression],
            A:Union[pmo.variable, pmo.expression],
            alphaPrime:float,
            alphaMin:float,
            alphaMax:float,
            aRelationOption:ARelationOption=ARelationOption.GEQ,
            epsilon:float=1e-6,
        ):
        super().__init__(
            ["X", "Y", "A"],
            {
                "X": (X, (0, 1)),
                "Y": (Y, (0, 1)),
                "A": (A, (alphaMin, alphaMax))
            }
        )

        self.alphaPrime = alphaPrime
        self.alphaMin = alphaMin
        self.alphaMax = alphaMax
        if epsilon <= 0:
            warn(f"Epsilon value {epsilon} is non-positive. This may lead to numerical issues in the formulation. For virtually all usages, epsilon should be a small positive value (e.g., 1e-6).", UserWarning)
        self.epsilon = epsilon

        if aRelationOption == ARelationOption.GEQ:
            self._initGEQ(X, Y, A)
        elif aRelationOption == ARelationOption.LEQ:
            self._initLEQ(X, Y, A)
        else:
            raise ValueError(f"Invalid ARelationOption: {aRelationOption}. Must be one of {list(ARelationOption)}")
        
    def computePlane(self, points):
        """
        Computes the plane defined by three points in 3D space.

        Planes or of the form:

            C1 * X + C2 * Y + C3 * A = C4

        where C1, C2, C3, and C4 are coefficients derived from the points.
        Parameters
        ----------
        points: list of tuples
            A list of three tuples, each containing the (X, Y, A) coordinates of a point defining the plane.
        """
        points = np.asarray(points)
    
        v1 = points[1] - points[0]  
        v2 = points[2] - points[0] 
        normal = np.cross(v1, v2)  
        
        d = np.dot(normal, points[0])
        
        return np.array([normal[0], normal[1], normal[2], d])
    
    def _constructConstraint(self, X, Y, A, points, violatingPoint, name=None) -> pmo.constraint:
        """
        A function to construct a constraint based on the points defining a plane and a violating point.

        Parameters
        ----------
        X: pmo.variable | pmo.expression
            The Pyomo variable or expression representing "X" in this relationship.
        Y: pmo.variable | pmo.expression
            The Pyomo variable or expression representing "Y" in this relationship.
        A: pmo.variable | pmo.expression
            The Pyomo variable or expression representing "A" in this relationship.
        points: list of tuples
            A list of three tuples, each containing the (X, Y, A) coordinates of a point defining the plane.
        violatingPoint: tuple
            A tuple containing the (X, Y, A) coordinates of a point that violates the desired constraint.
        name: str (optional)
            The name of the constraint to be created. If None, a default name will be generated.
        """
        C = self.computePlane(points)

        predictedValue = C[:3] @ np.array(violatingPoint)
        if predictedValue < C[3]:
            self.registerConstraint(
                lambda X, Y, A, c=C: c[0] * X + c[1] * Y + c[2] * A >= c[3],
            )
        else:
            self.registerConstraint(
                lambda X, Y, A, c=C: c[0] * X + c[1] * Y + c[2] * A <= c[3],
            )

        #TODO: Finish converting other formuations over to the _Formulation object. Then keep debugging this.


    def _initGEQ(self, X, Y, A):
        self._constructConstraint(
            X, Y, A,
            [
                (0,0,self.alphaMin), 
                (0,1,self.alphaMin), 
                (1,1,self.alphaPrime)
            ],
            (1,1,self.alphaMin)
        )
        self._constructConstraint(
            X, Y, A,
            [
                (0,0,self.alphaMin), 
                (1,1,self.alphaMin), 
                (0,0,self.alphaMax)
            ],
            (1,0,self.alphaMin)
        )
        self._constructConstraint(
            X, Y, A,
            [
                (0,1,self.alphaPrime-self.epsilon), 
                (0,0,self.alphaMax), 
                (1,1,self.alphaMax)
            ],
            (0,1,self.alphaMax)
        )


    def _initLEQ(self, X, Y, A):
        self._constructConstraint(
            X, Y, A,
            [
                (0,0,self.alphaMin), 
                (0,1,self.alphaPrime+self.epsilon), 
                (1,1,self.alphaMin)
            ],
            (0,1,self.alphaMin)
        )
        self._constructConstraint(
            X, Y, A,
            [
                (0,0,self.alphaMin), 
                (1,1,self.alphaMin), 
                (0,0,self.alphaMax)
            ],
            (1,0,self.alphaMin)
        )
        self._constructConstraint(
            X, Y, A,
            [
                (0,1,self.alphaMax), 
                (1,1,self.alphaPrime), 
                (0,0,self.alphaMax)
            ],
            (1,1,self.alphaMax)
        )

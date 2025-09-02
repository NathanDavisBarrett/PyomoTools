import pyomo.environ as pyo

from .VectorRepresentation.VectorRepresentation import VectorRepresentation

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import numpy as np
from scipy.spatial import ConvexHull as scipy_ConvexHull
from scipy.sparse import csr_matrix, vstack, hstack


class Polytope:
    """
    A class to house the analysis of a given polytope within a linear model.
    """

    def __init__(self, model: pyo.ConcreteModel, vars: list):
        """
        Parameters
        ----------
        model: pyo.ConcreteModel
            The model containing the polytope you'd like to analyze.
        vars: list
            A list of pyomo variable objects corresponding with the variables present in the polytope you'd like to analyze.

        """
        # Ensure that each of the key variables you're plotting have bounds:
        for var in vars:
            lb, ub = var.bounds
            assert lb is not None, f"Key Variable {var} has now lower bound."
            assert ub is not None, f"Key Variable {var} has now upper bound."

        # Construct vector representation.
        self.vecRep = VectorRepresentation(model)

        # Construct model-wide matrix representation.
        A, b, c, d, inequalityIndices, equalityIndices = (
            self.vecRep.Generate_Matrix_Representation()
        )

        # Select only the constraints (rows) relevant to the variables indicated.
        self.targetVarNames = [str(v) for v in vars]

        varIndices = np.array([self.vecRep.VAR_VEC_INV[v] for v in self.targetVarNames])

        relevantConstrIndices = self.GetRelevantConstraints(A, varIndices)

        allVarNames = np.array([str(v) for v in self.vecRep.VAR_VEC])
        allConstrNames = np.array([str(c) for c in self.vecRep.CONSTR_VEC])

        relevantInequalityIndices = np.intersect1d(
            inequalityIndices, relevantConstrIndices
        )
        relevantEqualityIndices = np.intersect1d(equalityIndices, relevantConstrIndices)

        A_leq = A[relevantInequalityIndices, :]
        b_leq = b[relevantInequalityIndices]
        A_eq = A[relevantEqualityIndices, :]
        b_eq = b[relevantEqualityIndices]

        relevantInequalityNames = allConstrNames[relevantInequalityIndices]
        relevantEqualityNames = allConstrNames[relevantEqualityIndices]

        A_intermed = vstack([A_leq, A_eq, -A_eq])
        self.b_final = hstack([b_leq, b_eq, -b_eq])

        self.relevantConstrNames = np.hstack(
            [relevantInequalityNames, relevantEqualityNames, relevantEqualityNames]
        )

        # Drop any columns (variables) that are not relevant to this sub-set of constraints.
        columnSums = np.asarray(abs(A_intermed).sum(axis=0)).flatten()
        nonZeroColumnIndices = np.where(columnSums > 0)[0]

        A_itermed = A_intermed[:, nonZeroColumnIndices]
        relevantVarNames = allVarNames[nonZeroColumnIndices]

        # Now split "A" into "A_final" and "A_known" for the final form of this polytope:
        #   A_final x_target <= b_final - A_known x_known
        #   Where x_known are all the variables you'll need to specify values for in order to evaluate this polytope.
        targetVarIndices = np.array(
            [np.where(relevantVarNames == str(v))[0] for v in vars]
        ).flatten()
        mask = np.ones(len(relevantVarNames), dtype=bool)
        mask[targetVarIndices] = False
        knownVarIndices = np.arange(len(relevantVarNames), dtype=int)[mask]

        self.A_final = A_itermed[:, targetVarIndices].toarray()
        self.A_known = A_intermed[:, knownVarIndices].toarray()
        self.knownVarNames = relevantVarNames[mask]

    def GetRelevantConstraints(self, A: csr_matrix, varIndices: np.array):
        """
        A function to collect the constraints (rows of the A matrix) that involve any of the variables mentioned.

        Parameters
        ----------
        vecRep: VectorRepresentation
            The vector representation of the model containing the variables and constraints you'd like to collect.
        varIndices: numpy array or int
            An array containing the indices of the variables you'd like to consider.

        Returns
        -------
        constrIndices: np.array of int
            An array of the constraint indices of the A matrix that correspond with any of the provided variables indices
        """
        coo = A.tocoo()

        mask = np.isin(coo.col, varIndices)
        nonzeroRows = np.unique(coo.row[mask])

        return nonzeroRows

    def Evaluate_b(self, knownValues=None):
        """
        A function to compute the terminal representation of the "b" vector.

        knownValues: dict (optional, Default = None)
            A dict mapping the name of each of the nececcary additional variables (see self.knownVarNames) to the value of that variable. If None is provided, the values will be taken directly from the model initially provided to this Polytope object.

        Returns
        -------
        b: np.ndarray
            The terminal representation of the "b" vector.
        """
        if knownValues is not None:
            x_known = np.array([knownValues[v] for v in self.knownVarNames])
        else:
            x_known = np.empty(len(self.knownVarNames), dtype=float)
            for i, v in enumerate(self.knownVarNames):
                var = self.vecRep.VAR_VEC[self.vecRep.VAR_VEC_INV[v]]
                x_known[i] = pyo.value(var)

        b = self.b_final - self.A_known @ x_known
        return np.asarray(b).flatten()

    def EnumerateVertices(self, b):
        """
        A function to evaluate the vertices of this polytope.

        b: np.ndarray
            The b vector you'd like to use in representation of the polytope.

        Returns
        -------
        vertices: np.array
            A numpy array of shape (m,n) where n is the number of variables involved in this polytope (initially provided in the init function) and m in the number of feasible vertices.
        """
        from pypoman import compute_polytope_vertices

        return np.array(compute_polytope_vertices(self.A_final, b))

    def ConvexHull(self, vertices):
        """
        A function to generate the indices of each face of a polyhedron from an array of the vertices of that polyhedron.

        Parameters
        ----------
        vertices: np.array
            An array of shape (m,n) where m in the number of vertices of the polyhedron and n is the dimensionality of the polyhedron

        Returns
        -------
        faceIndices: list
            A list of tuples where each tuple represents on face. Each element of each tuple represents the index of each vertex in that face.
        """
        allFaces = scipy_ConvexHull(vertices).simplices

        allFaces = [tuple(allFaces[i]) for i in range(len(allFaces))]

        seenCs = []
        seenBs = []
        seenSets = []

        for face in allFaces:
            A = vertices[face, :]
            m, n = A.shape

            A = np.hstack((A, np.ones((m, 1))))
            b = np.ones(m)
            coefs = np.linalg.lstsq(A, b)[0]

            c = coefs[:-1]
            b = coefs[-1]

            # Check to see if plane overlaps with any of the planes we've seen so far.
            matchFound = False
            for i in range(len(seenCs)):
                cp = seenCs[i]
                bp = seenBs[i]
                isMatch = True
                if np.allclose([bp], [0]):
                    if np.allclose([b], [0]):
                        cpp = cp
                    else:
                        isMatch = False
                else:
                    cpp = cp * b / bp

                if isMatch:
                    isMatch = np.allclose(c, cpp)

                if isMatch:
                    # Add these indices to those belonging to this set.
                    seenSets[i].update(face)
                    matchFound = True
                    break

            if not matchFound:
                seenCs.append(c)
                seenBs.append(b)
                seenSets.append(set(face))

        seenSets = [tuple(s) for s in seenSets]

        # Now sort the indices so that they plot correctly.
        for i in range(len(seenSets)):
            indices = seenSets[i]
            theseVertices = vertices[indices, :]

            centroid = np.sum(theseVertices, axis=0) / len(theseVertices)
            centeredVertices = theseVertices - centroid

            U, S, Vt = np.linalg.svd(centeredVertices)

            twoDPoints = np.dot(centeredVertices, Vt.T[:, :2])

            twoD_Center = np.sum(twoDPoints, axis=0) / len(twoDPoints)

            twoDPoints -= twoD_Center

            twoDAngles = np.arctan2(twoDPoints[:, 0], twoDPoints[:, 1])

            sortData = [[indices[j], twoDAngles[j]] for j in range(len(indices))]
            sortData.sort(key=lambda x: x[1])

            seenSets[i] = tuple([sortData[j][0] for j in range(len(indices))])

        return seenSets

    def Plot(self):
        """
        A function to plot this polytope
        """
        # Detect of 2D or 3D
        self.numDim = len(self.targetVarNames)

        if self.numDim not in [2, 3]:
            raise Exception(f"Unable to plot polytopes of dimension {self.numDim}.")

        # Make a plot
        self.fig = plt.figure()
        if self.numDim == 2:
            self.ax = self.fig.add_subplot(111)
        else:
            self.ax = self.fig.add_subplot(111, projection="3d")

        # Adjust to make room for the sliders:
        self.fig.subplots_adjust(left=0.25)

        # Add sliders for each known variable.
        self.sliders = []

        for i, v in enumerate(self.knownVarNames):
            var = self.vecRep.VAR_VEC[self.vecRep.VAR_VEC_INV[v]]
            bounds = var.bounds
            sliderAx = self.fig.add_axes(
                [0.05, i / len(self.knownVarNames), 0.1, 1 / len(self.knownVarNames)]
            )
            val = var.value if var.value is not None else (bounds[1] + bounds[0]) / 2
            slider = Slider(
                ax=sliderAx, label=v, valmin=bounds[0], valmax=bounds[1], valinit=val
            )
            if self.numDim == 2:
                slider.on_changed(self._UpdatePlot_2D)
            elif self.numDim == 3:
                slider.on_changed(self._UpdatePlot_3D)
            self.sliders.append(slider)

        # Initialize each plot
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        self.defaultColorCycle = prop_cycle.by_key()["color"]
        self.constrPlots = [None for _ in range(len(self.relevantConstrNames))]
        self.constrIntersectLines = {}
        if self.numDim == 2:
            self.vertexScatter = self.ax.scatter(
                [
                    0,
                ],
                [
                    0,
                ],
                marker="o",
            )

            for i in range(len(self.relevantConstrNames)):
                (self.constrPlots[i],) = self.ax.plot(
                    [
                        0,
                    ],
                    [
                        0,
                    ],
                    label=self.relevantConstrNames[i],
                )
        else:
            for i in range(len(self.relevantConstrNames)):
                junk = np.array([0, 1])
                junkX, junkY = np.meshgrid(junk, junk)
                self.constrPlots[i] = self.ax.plot_surface(
                    junkX, junkX, junkX, alpha=0.5, color=self.defaultColorCycle[i]
                )

                for ip in range(i + 1, len(self.relevantConstrNames)):
                    (self.constrIntersectLines[i, ip],) = self.ax.plot(
                        [
                            0,
                        ],
                        [
                            0,
                        ],
                        [
                            0,
                        ],
                        "k",
                    )

                self.ax.plot(
                    [
                        0,
                    ],
                    [
                        0,
                    ],
                    [
                        0,
                    ],
                    color=self.defaultColorCycle[i],
                    alpha=0.5,
                    label=self.relevantConstrNames[i],
                )

            self.ax.set_xlabel(self.targetVarNames[0])
            self.ax.set_ylabel(self.targetVarNames[1])
            self.ax.set_zlabel(self.targetVarNames[2])

        if self.numDim == 2:
            self._UpdatePlot_2D()
        elif self.numDim == 3:
            self._UpdatePlot_3D()

        self.ax.legend()

        plt.show()

    def _GetConstrData_2D(self, i, b, mins, maxs):
        Ai = self.A_final[i, :]
        bi = b[i]

        # Check if vertical line:
        if Ai[1] == 0:
            xVal = bi / Ai[0]
            return np.array([xVal, xVal]), np.array([mins[1], maxs[1]])
        else:
            xs = np.array([mins[0], maxs[0]])
            ys = (bi - Ai[0] * xs) / Ai[1]
            return xs, ys

    def _UpdatePlot_2D(self, *junk):
        knownVals = {v: self.sliders[i].val for i, v in enumerate(self.knownVarNames)}

        b = self.Evaluate_b(knownVals)
        vertices = self.EnumerateVertices(b)
        mins = np.min(vertices, axis=0)
        maxs = np.max(vertices, axis=0)

        deltas = (maxs - mins) / 10
        mins -= deltas
        maxs += deltas

        self.vertexScatter.set_offsets(np.c_[vertices[:, 0], vertices[:, 1]])

        xs = np.linspace(mins[0], maxs[0], 200)
        ys = np.linspace(mins[1], maxs[1], 200)
        X, Y = np.meshgrid(xs, ys)
        fesibleMask = np.ones_like(X, dtype=bool)

        for i in range(len(self.relevantConstrNames)):
            xdata, ydata = self._GetConstrData_2D(i, b, mins, maxs)
            self.constrPlots[i].set_xdata(xdata)
            self.constrPlots[i].set_ydata(ydata)

            fesibleMask = np.logical_and(
                fesibleMask, self.A_final[i, 0] * X + self.A_final[i, 1] * Y <= b[i]
            )

        if hasattr(self, "feasibleRegionPlot"):
            self.feasibleRegionPlot.remove()
        self.feasibleRegionPlot = self.ax.pcolormesh(
            X,
            Y,
            fesibleMask,
            cmap="Blues",
            shading="auto",
            alpha=0.5,
            label="Feasible Region",
        )

        self.ax.set_xlim([mins[0], maxs[0]])
        self.ax.set_ylim([mins[1], maxs[1]])

        self.fig.canvas.draw_idle()

    def _GetConstrData_3D(self, i, b, mins, maxs):
        Ai = self.A_final[i, :]
        bi = b[i]

        # It's best to leave the computed values as small as possible.
        axisRanking = [[i, np.abs(Ai[i])] for i in range(3)]
        axisRanking.sort(key=lambda x: x[1])

        axisRanking = [x[0] for x in axisRanking]

        if Ai[axisRanking[2]] == 0:
            if Ai[axisRanking[1]] == 0:
                if Ai[axisRanking[0]] == 0:
                    raise Exception("Cannot plot plane 0 == 0")
                X1, X2 = np.meshgrid(
                    np.array([mins[axisRanking[1]], maxs[axisRanking[1]]]),
                    np.array([mins[axisRanking[2]], maxs[axisRanking[2]]]),
                )
                X0 = np.full_like(X1, bi / Ai[axisRanking[0]])
            else:
                X0, X2 = np.meshgrid(
                    np.array([mins[axisRanking[0]], maxs[axisRanking[0]]]),
                    np.array([mins[axisRanking[2]], maxs[axisRanking[2]]]),
                )
                X1 = (-X0 * Ai[axisRanking[0]] + bi) / Ai[axisRanking[1]]
        else:
            X0, X1 = np.meshgrid(
                np.linspace(mins[axisRanking[0]], maxs[axisRanking[0]], 2),
                np.linspace(mins[axisRanking[1]], maxs[axisRanking[1]], 2),
            )
            X2 = (-X0 * Ai[axisRanking[0]] - X1 * Ai[axisRanking[1]] + bi) / Ai[
                axisRanking[2]
            ]
            # X2[X2<[mins[axisRanking[2]]]] = np.nan
            # X2[X2>[maxs[axisRanking[2]]]] = np.nan

        if axisRanking[0] == 0:
            X = X0
            if axisRanking[1] == 1:
                Y = X1
                Z = X2
            else:
                Y = X2
                Z = X1
        elif axisRanking[0] == 1:
            Y = X0
            if axisRanking[1] == 0:
                X = X1
                Z = X2
            else:
                X = X2
                Z = X1
        else:
            Z = X0
            if axisRanking[1] == 0:
                X = X1
                Y = X2
            else:
                X = X2
                Y = X1

        return X, Y, Z

    def _UpdatePlot_3D(self, *junk):
        knownVals = {v: self.sliders[i].val for i, v in enumerate(self.knownVarNames)}

        b = self.Evaluate_b(knownVals)
        vertices = self.EnumerateVertices(b)
        mins = np.min(vertices, axis=0)
        maxs = np.max(vertices, axis=0)

        deltas = (maxs - mins) / 10
        mins -= deltas
        maxs += deltas

        xs = np.linspace(mins[0], maxs[0], 200)
        ys = np.linspace(mins[1], maxs[1], 200)
        zs = np.linspace(mins[2], maxs[2], 200)
        X, Y, Z = np.meshgrid(xs, ys, zs)
        fesibleMask = np.ones_like(X, dtype=bool)

        # self.ax.collections.clear()
        for i in range(len(self.relevantConstrNames)):
            Xi, Yi, Zi = self._GetConstrData_3D(i, b, mins, maxs)

            self.constrPlots[i].remove()
            self.constrPlots[i] = self.ax.plot_surface(
                Xi, Yi, Zi, alpha=0.5, color=self.defaultColorCycle[i]
            )
            fesibleMask = np.logical_and(
                fesibleMask,
                self.A_final[i, 0] * X + self.A_final[i, 1] * Y + self.A_final[i, 2] * Z
                <= b[i],
            )

            for ip in range(i + 1, len(self.relevantConstrNames)):
                Acomb = np.vstack([self.A_final[i, :], self.A_final[ip, :]])
                bcomb = np.array([b[i], b[ip]])
                directionVector = np.cross(self.A_final[i, :], self.A_final[ip, :])
                baseSolution = np.linalg.lstsq(Acomb, bcomb, rcond=None)[0]
                tmin = (mins - baseSolution) / directionVector
                tmax = (maxs - baseSolution) / directionVector

                for j in range(3):
                    if np.isnan(tmin[j]) or np.isnan(tmax[j]):
                        continue
                    if tmin[j] >= tmax[j]:
                        temp = tmin[j]
                        tmin[j] = tmax[j]
                        tmax[j] = temp

                tmin = np.max(tmin)
                tmax = np.min(tmax)

                start = baseSolution + tmin * directionVector
                stop = baseSolution + tmax * directionVector

                if self.constrIntersectLines[i, ip] is not None:
                    self.constrIntersectLines[i, ip].remove()

                if tmin < tmax:
                    (self.constrIntersectLines[i, ip],) = self.ax.plot(
                        *[[start[i], stop[i]] for i in range(3)], "k"
                    )
                else:
                    self.constrIntersectLines[i, ip] = None

        # if hasattr(self,"feasibleRegionPlot"):
        #     self.feasibleRegionPlot.remove()
        # self.feasibleRegionPlot = self.ax.pcolormesh(X,Y,fesibleMask, cmap='Blues',shading='auto',alpha=0.5,label="Feasible Region")

        self.ax.set_xlim([mins[0], maxs[0]])
        self.ax.set_ylim([mins[1], maxs[1]])
        self.ax.set_zlim([mins[1], maxs[1]])

        self.fig.canvas.draw_idle()

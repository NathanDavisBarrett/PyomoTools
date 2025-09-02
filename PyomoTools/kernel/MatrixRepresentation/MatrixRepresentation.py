import pyomo.kernel as pmo
from pyomo.core.expr.numeric_expr import (
    MonomialTermExpression,
    NegationExpression,
    ProductExpression,
    SumExpression,
)
from pyomo.core.expr.numvalue import NumericConstant

from scipy.sparse import csr_matrix
import numpy as np

from functools import cached_property


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
            yield thisNode.val
            thisNode = thisNode.next

    def iterval(self):
        thisNode = self.headNode
        while thisNode is not None:
            yield thisNode.val
            thisNode = thisNode.next


class MatrixRepresentation:
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

    def __init__(self, model: pmo.block):
        self.model = model

        self.VAR_VEC = list(self.gatherVariables(model))
        self.VAR_VEC_INV = {str(var): i for i, var in enumerate(self.VAR_VEC)}

        constraints = list(self.gatherConstraints(model))
        self.CONSTR_VEC = [c for c in constraints]
        self.CONSTR_VEC_INV = {str(c): i for i, c in enumerate(constraints)}

        self.A, self.b, self.equalityIndices, self.inequalityIndices = self.AssembleAb()
        self.c, self.d = self.AssembleObjective()

    def _renderLinearEq(self, A, b, x, operator="="):
        Astr = str(A)
        Arows = Astr.split("\n")
        maxAWidth = max([len(row) for row in Arows])
        for i in range(len(Arows)):
            Arows[i] = Arows[i].ljust(maxAWidth)

        b = b.reshape((len(Arows), 1))
        brows = str(b).split("\n")
        centerIndex = len(Arows) // 2

        xRows = str(np.array([str(xi) for xi in x]).reshape((len(x), 1))).split("\n")
        maxXwidth = max([len(xi) for xi in xRows])
        for i in range(len(xRows)):
            xRows[i] = xRows[i].ljust(maxXwidth)

        allRows = []
        for i in range(len(Arows)):
            op = operator if i == centerIndex else " " * len(operator)
            xRow = xRows[i] if i < len(x) else " " * maxXwidth
            allRows.append(Arows[i] + xRow + op + brows[i])
        for i in range(len(Arows), len(x)):
            allRows.append(" " * maxAWidth + str(xRows[i]).ljust(maxXwidth))

        return "\n".join(allRows)

    @cached_property
    def Aeq(self):
        """
        A property to return the equality constraints in matrix form.
        """
        return self.A[self.equalityIndices, :]

    @cached_property
    def Beq(self):
        """
        A property to return the equality constraints in vector form.
        """
        return self.b[self.equalityIndices]

    @cached_property
    def Aleq(self):
        """
        A property to return the inequality constraints in matrix form.
        """
        return self.A[self.inequalityIndices, :]

    @cached_property
    def Bleq(self):
        """
        A property to return the inequality constraints in vector form.
        """
        return self.b[self.inequalityIndices]

    def __str__(self):
        varNames = [str(var) for var in self.VAR_VEC]

        ans = []
        ans.append(
            "min [" + ",".join(self.c.astype(str)) + "]^T [" + ",".join(varNames) + "]"
        )
        ans.append("subject to:")
        A = np.array(self.A.todense())
        b = self.b

        A_eq = A[self.equalityIndices, :]
        b_eq = b[self.equalityIndices]
        A_leq = A[self.inequalityIndices, :]
        b_leq = b[self.inequalityIndices]

        if len(self.equalityIndices) > 0:
            ans.append(self._renderLinearEq(A_eq, b_eq, varNames, "="))
            ans.append("")
        if len(self.inequalityIndices) > 0:
            ans.append(self._renderLinearEq(A_leq, b_leq, varNames, "<="))
            ans.append("")

        return "\n".join(ans)

    def computeAllVertices(self):
        return self.vertices

    def testSolution(self, solution):
        b_eq_pred = self.Aeq @ solution
        if not np.allclose(b_eq_pred, self.Beq):
            return False

        b_leq_pred = self.Aleq @ solution
        if not np.all(b_leq_pred <= self.Bleq + 1e-6):
            return False

        return True

    @cached_property
    def vertices(self):
        from itertools import combinations

        dim = len(self.VAR_VEC)
        numConstr = len(self.CONSTR_VEC)
        A = self.A.todense()
        b = self.b

        if dim > numConstr:
            return (
                []
            )  # There cannot be any vertices if there are more variables than constraints.

        vertices = LinkedList([])

        comb = combinations(range(numConstr), dim)

        for constrIndices in comb:
            constrIndices = np.array(constrIndices)
            subA = A[constrIndices, :]
            subb = b[constrIndices]

            try:
                solution = np.linalg.solve(subA, subb)
            except np.linalg.LinAlgError:
                continue  # This set of constraints does not have a solution.
            if self.testSolution(solution):
                vertices.append(solution)

        return np.array([vertex for vertex in vertices], dtype=float)

    def _Plot1D(self, ax=None):
        import matplotlib.pyplot as plt

        mn = self.vertices.min()
        mx = self.vertices.max()

        ownership = ax is None

        if ownership:
            fig, ax = plt.subplots(1, 1)
        ax.axhline(0, color="black", lw=0.5)

        ax.fill_between([mn, mx], [-1, -1], [1, 1], color="blue")
        ax.set_ylim([-4, 4])
        ax.set_yticks([])
        ax.set_xlabel(self.VAR_VEC[0].name)

        if ownership:
            plt.show()
        return ax

    def _Plot2D(self, ax=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        ownership = ax is None
        if ownership:
            fig, ax = plt.subplots(1, 1)
        p = Polygon(self.vertices, closed=True, fill=True, color="blue")
        ax.add_patch(p)

        ax.set_xlabel(self.VAR_VEC[0].name)
        ax.set_ylabel(self.VAR_VEC[1].name)

        if ownership:
            plt.show()

        return ax

    def _Plot3D(self, ax=None):
        import matplotlib.pyplot as plt
        from itertools import combinations
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        ownership = ax is None
        if ownership:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        # For 3D polytope, we need to compute the convex hull
        from scipy.spatial import ConvexHull
        from scipy.spatial._qhull import QhullError

        try:

            if self.vertices.shape[1] == 3:  # 3D points
                hull = ConvexHull(self.vertices)
                for simplex in hull.simplices:
                    verts = self.vertices[simplex]
                    poly = Poly3DCollection(
                        [verts],
                        facecolors="lightblue",
                        linewidths=1,
                        edgecolors="blue",
                        alpha=0.6,
                    )
                    ax.add_collection3d(poly)
        except QhullError:
            # Fallback: just show triangular faces between vertices
            comb = combinations(range(len(self.vertices)), 3)
            for i, vcomb in enumerate(comb):
                verts = self.vertices[list(vcomb), :]
                poly = Poly3DCollection(
                    [verts],
                    facecolors="lightblue",
                    linewidths=1,
                    edgecolors="blue",
                    alpha=0.4,
                )
                ax.add_collection3d(poly)

        ax.scatter(
            self.vertices[:, 0],
            self.vertices[:, 1],
            self.vertices[:, 2],
            c="red",
            s=50,
            alpha=0.8,
        )

        ax.set_xlabel(self.VAR_VEC[0].name)
        ax.set_ylabel(self.VAR_VEC[1].name)
        ax.set_zlabel(self.VAR_VEC[2].name)

        if ownership:
            plt.show()

        return ax

    def _PlotHigherDim(self, ax=None):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, CheckButtons
        from itertools import combinations
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np

        if ax is not None:
            raise ValueError(
                "_PlotHigherDim does not support external axes due to widget requirements"
            )

        numVar = len(self.VAR_VEC)

        # Initialize default axis assignments (first 3 variables)
        x_axis_var = 0
        y_axis_var = 1 if numVar > 1 else 0
        z_axis_var = 2 if numVar > 2 else 0

        # Initialize slider values at midpoint of variable ranges
        slider_values = np.zeros(numVar)
        if len(self.vertices) > 0:
            var_mins = np.min(self.vertices, axis=0)
            var_maxs = np.max(self.vertices, axis=0)
            slider_values = (var_mins + var_maxs) / 2

        # Create figure with custom layout
        fig = plt.figure(figsize=(14, 10))

        # Main 3D plot - takes up most of the space
        ax_3d = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=2, projection="3d")

        # Checkbox areas for axis selection
        ax_x_check = plt.subplot2grid((4, 3), (2, 0))
        ax_y_check = plt.subplot2grid((4, 3), (2, 1))
        ax_z_check = plt.subplot2grid((4, 3), (2, 2))

        # Create variable name lists for checkboxes
        var_names = [var.name for var in self.VAR_VEC]

        # Create checkboxes for axis selection
        x_checks = CheckButtons(
            ax_x_check, var_names, [i == x_axis_var for i in range(numVar)]
        )
        y_checks = CheckButtons(
            ax_y_check, var_names, [i == y_axis_var for i in range(numVar)]
        )
        z_checks = CheckButtons(
            ax_z_check, var_names, [i == z_axis_var for i in range(numVar)]
        )

        ax_x_check.set_title("X-Axis Variable", fontsize=10)
        ax_y_check.set_title("Y-Axis Variable", fontsize=10)
        ax_z_check.set_title("Z-Axis Variable", fontsize=10)

        # Storage for sliders and current state
        sliders = []
        updating_checkboxes = False  # Flag to prevent infinite callback loops

        def create_sliders():
            """Create sliders for variables not currently displayed on axes"""
            nonlocal sliders

            # Clear existing sliders
            for slider in sliders:
                if hasattr(slider, "ax"):
                    slider.ax.remove()
            sliders = []

            # Determine which variables need sliders
            axis_vars = {x_axis_var, y_axis_var, z_axis_var}
            slider_vars = [i for i in range(numVar) if i not in axis_vars]

            if len(slider_vars) == 0:
                return

            # Calculate slider layout
            slider_height = 0.03
            slider_spacing = 0.04
            start_bottom = 0.08

            if len(self.vertices) > 0:
                var_mins = np.min(self.vertices, axis=0)
                var_maxs = np.max(self.vertices, axis=0)
            else:
                var_mins = np.zeros(numVar)
                var_maxs = np.ones(numVar)

            for i, var_idx in enumerate(slider_vars):
                # Position slider
                slider_bottom = start_bottom + i * slider_spacing
                ax_slider = fig.add_axes([0.15, slider_bottom, 0.7, slider_height])

                # Create slider
                var_min = var_mins[var_idx]
                var_max = var_maxs[var_idx]
                if abs(var_max - var_min) < 1e-10:
                    var_max = var_min + 1  # Avoid zero-width range

                slider = Slider(
                    ax_slider,
                    var_names[var_idx],
                    var_min,
                    var_max,
                    valinit=slider_values[var_idx],
                )

                # Create closure to capture var_idx
                def make_slider_callback(variable_index):
                    def slider_callback(val):
                        slider_values[variable_index] = val
                        update_plot()

                    return slider_callback

                slider.on_changed(make_slider_callback(var_idx))
                sliders.append(slider)

        def update_plot():
            """Update the 3D plot based on current selections"""
            ax_3d.clear()

            # Get current axis variable indices
            axis_vars = [x_axis_var, y_axis_var, z_axis_var]
            non_axis_vars = [i for i in range(numVar) if i not in axis_vars]

            if len(self.vertices) == 0:
                ax_3d.text(
                    0.5,
                    0.5,
                    0.5,
                    "No feasible vertices found",
                    transform=ax_3d.transAxes,
                    ha="center",
                    va="center",
                )
                return

            # For higher dimensional polytopes, we'll show the projection
            # and optionally filter vertices that are close to slider values

            # Project all vertices to the selected 3D space
            projected_vertices = []
            for vertex in self.vertices:
                projected_vertex = [
                    vertex[x_axis_var],
                    vertex[y_axis_var],
                    vertex[z_axis_var],
                ]
                projected_vertices.append(projected_vertex)

            projected_vertices = np.array(projected_vertices)

            # Remove duplicate projected vertices
            unique_vertices = []
            tolerance = 1e-8
            for vertex in projected_vertices:
                is_duplicate = False
                for existing in unique_vertices:
                    if np.linalg.norm(vertex - existing) < tolerance:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_vertices.append(vertex)

            if len(unique_vertices) > 0:
                unique_vertices = np.array(unique_vertices)

                # Create 3D polytope visualization
                if len(unique_vertices) >= 4:
                    # For 3D polytope, we need to compute the convex hull
                    from scipy.spatial import ConvexHull
                    from scipy.spatial._qhull import QhullError

                    try:
                        if unique_vertices.shape[1] == 3:  # 3D points
                            hull = ConvexHull(unique_vertices)
                            for simplex in hull.simplices:
                                verts = unique_vertices[simplex]
                                poly = Poly3DCollection(
                                    [verts],
                                    facecolors="lightblue",
                                    linewidths=1,
                                    edgecolors="blue",
                                    alpha=0.6,
                                )
                                ax_3d.add_collection3d(poly)
                    except QhullError:
                        # Fallback: just show triangular faces between vertices
                        comb = combinations(range(len(unique_vertices)), 3)
                        for i, vcomb in enumerate(comb):
                            verts = unique_vertices[list(vcomb), :]
                            poly = Poly3DCollection(
                                [verts],
                                facecolors="lightblue",
                                linewidths=1,
                                edgecolors="blue",
                                alpha=0.4,
                            )
                            ax_3d.add_collection3d(poly)

                # Plot vertices as points
                ax_3d.scatter(
                    unique_vertices[:, 0],
                    unique_vertices[:, 1],
                    unique_vertices[:, 2],
                    c="red",
                    s=50,
                    alpha=0.8,
                )

            # Set axis labels
            ax_3d.set_xlabel(var_names[x_axis_var])
            ax_3d.set_ylabel(var_names[y_axis_var])
            ax_3d.set_zlabel(var_names[z_axis_var])

            # Set axis limits based on projected vertices
            if len(unique_vertices) > 0:
                margin = 0.1
                x_range = np.max(unique_vertices[:, 0]) - np.min(unique_vertices[:, 0])
                y_range = np.max(unique_vertices[:, 1]) - np.min(unique_vertices[:, 1])
                z_range = np.max(unique_vertices[:, 2]) - np.min(unique_vertices[:, 2])

                ax_3d.set_xlim(
                    [
                        np.min(unique_vertices[:, 0]) - margin * x_range,
                        np.max(unique_vertices[:, 0]) + margin * x_range,
                    ]
                )
                ax_3d.set_ylim(
                    [
                        np.min(unique_vertices[:, 1]) - margin * y_range,
                        np.max(unique_vertices[:, 1]) + margin * y_range,
                    ]
                )
                ax_3d.set_zlim(
                    [
                        np.min(unique_vertices[:, 2]) - margin * z_range,
                        np.max(unique_vertices[:, 2]) + margin * z_range,
                    ]
                )

            # Add text showing current slider values for non-displayed variables
            info_text = []
            non_axis_vars = [i for i in range(numVar) if i not in axis_vars]
            for var_idx in non_axis_vars:
                info_text.append(f"{var_names[var_idx]}: {slider_values[var_idx]:.3f}")

            if info_text:
                ax_3d.text2D(
                    0.02,
                    0.98,
                    "Fixed values:\n" + "\n".join(info_text),
                    transform=ax_3d.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                )

            fig.canvas.draw()

        def on_x_check(label):
            nonlocal x_axis_var, updating_checkboxes

            # Prevent infinite loop
            if updating_checkboxes:
                return

            # Find which variable was selected
            var_idx = var_names.index(label)

            # Only proceed if this is actually a change
            if var_idx == x_axis_var:
                return

            x_axis_var = var_idx

            # Set flag to prevent recursive calls
            updating_checkboxes = True

            # Uncheck all other boxes in this group
            for i, var_name in enumerate(var_names):
                if i != var_idx:
                    x_checks.set_active(i, False)

            # Clear flag
            updating_checkboxes = False

            create_sliders()
            update_plot()

        def on_y_check(label):
            nonlocal y_axis_var, updating_checkboxes

            # Prevent infinite loop
            if updating_checkboxes:
                return

            var_idx = var_names.index(label)

            # Only proceed if this is actually a change
            if var_idx == y_axis_var:
                return

            y_axis_var = var_idx

            # Set flag to prevent recursive calls
            updating_checkboxes = True

            for i, var_name in enumerate(var_names):
                if i != var_idx:
                    y_checks.set_active(i, False)

            # Clear flag
            updating_checkboxes = False

            create_sliders()
            update_plot()

        def on_z_check(label):
            nonlocal z_axis_var, updating_checkboxes

            # Prevent infinite loop
            if updating_checkboxes:
                return

            var_idx = var_names.index(label)

            # Only proceed if this is actually a change
            if var_idx == z_axis_var:
                return

            z_axis_var = var_idx

            # Set flag to prevent recursive calls
            updating_checkboxes = True

            for i, var_name in enumerate(var_names):
                if i != var_idx:
                    z_checks.set_active(i, False)

            # Clear flag
            updating_checkboxes = False

            create_sliders()
            update_plot()

        # Connect checkbox callbacks
        x_checks.on_clicked(on_x_check)
        y_checks.on_clicked(on_y_check)
        z_checks.on_clicked(on_z_check)

        # Create initial sliders and plot
        create_sliders()
        update_plot()

        plt.tight_layout()
        plt.show()

        return fig

    def Plot(self, ax=None):
        numVar = len(self.VAR_VEC)

        if numVar == 0:
            raise Exception("No variables found in this model. Cannot plot.")

        if numVar == 1:
            return self._Plot1D(ax=ax)
        elif numVar == 2:
            return self._Plot2D(ax=ax)
        elif numVar == 3:
            return self._Plot3D(ax=ax)
        else:
            return self._PlotHigherDim(ax=ax)

    def gatherVariables(self, block: pmo.block):
        vars = LinkedList([])
        for c in block.children():
            if isinstance(c, (pmo.variable_list, pmo.variable_tuple)):
                for i in range(len(c)):
                    vars.append(c[i])
            elif isinstance(c, pmo.variable_dict):
                for i in c:
                    vars.append(c[i])
            elif isinstance(c, pmo.variable):
                vars.append(c)

            elif isinstance(c, pmo.block):
                vars.extend(self.gatherVariables(c))
            elif isinstance(c, (pmo.block_list, pmo.block_tuple)):
                for i in range(len(c)):
                    vars.extend(self.gatherVariables(c[i]))
            elif isinstance(c, pmo.block_dict):
                for i in c:
                    vars.extend(self.gatherVariables(c[i]))
        return vars

    def variableBoundsToConstraints(self, var: pmo.variable):
        lbName = f"{var.name}_lb"
        lb = var.bounds[0]
        if lb is not None:
            var.lb = None
            setattr(self.model, lbName, pmo.constraint(lb <= var))
            lbc = getattr(self.model, lbName)
        else:
            lbc = None

        ubName = f"{var.name}_ub"
        ub = var.bounds[1]
        if ub is not None:
            var.ub = None
            setattr(self.model, ubName, pmo.constraint(var <= ub))
            ubc = getattr(self.model, ubName)
        else:
            ubc = None
        return lbc, ubc

    def gatherConstraints(self, block: pmo.block):
        for var in self.VAR_VEC:
            self.variableBoundsToConstraints(var)

        constrs = LinkedList([])
        for c in block.children():
            if isinstance(c, pmo.constraint):
                constrs.append(c)
            elif isinstance(c, (pmo.constraint_list, pmo.constraint_tuple)):
                for i in range(len(c)):
                    constrs.append(c[i])
            elif isinstance(c, pmo.constraint_dict):
                for i in c:
                    constrs.append(c[i])
            elif isinstance(c, pmo.block):
                constrs.extend(self.gatherConstraints(c))
            elif isinstance(c, (pmo.block_list, pmo.block_tuple)):
                for i in range(len(c)):
                    constrs.extend(self.gatherConstraints(c[i]))
            elif isinstance(c, pmo.block_dict):
                for i in c:
                    constrs.extend(self.gatherConstraints(c[i]))
        return constrs

    def AssembleAb(self):
        numConstr = len(self.CONSTR_VEC)
        numVar = len(self.VAR_VEC)
        A = LinkedList([])  # A linked list of [constrIndex,varIndex,value] lists

        b = np.empty(numConstr, dtype=float)
        equalityConstr = np.empty(numConstr, dtype=bool)

        for i, constr in enumerate(self.CONSTR_VEC):
            entries, const, equality = self._ParseConstraint(constr)
            for nodei in entries:
                A.append([i, *(nodei)])
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

        equalityIndices = np.where(equalityConstr)[0]

        inequalityIndices = np.where(np.logical_not(equalityConstr))[0]

        return A, b, equalityIndices, inequalityIndices

    def gatherObjective(self):
        objs = []
        for c in self.model.children():
            if isinstance(c, pmo.objective):
                objs.append(c)
            elif isinstance(c, (pmo.objective_list, pmo.objective_tuple)):
                for i in range(len(c)):
                    objs.append(c[i])
            elif isinstance(c, pmo.objective_dict):
                for i in c:
                    objs.append(c[i])
        return objs

    def AssembleObjective(self):
        objs = self.gatherObjective()

        numObj = len(objs)
        numVar = len(self.VAR_VEC)

        c = np.zeros(numVar, dtype=float)

        if numObj == 0:
            # No objective found
            # Do nothing
            d = 0
        elif numObj == 1:
            obj = objs[0]
            entries, d = self._ParseExpression(obj.expr)
            for varIndex, coef in entries.iterval():
                c[varIndex] = coef

            if obj.sense == pmo.maximize:
                c *= -1  # Always assume minimization.
        else:
            raise Exception(
                f"Currently, only one objective is supported. {numObj} were detected."
            )

        return c, d

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
            coli, vali = nodei
            for nodej in entries:
                colj = nodej[0]
                if coli == colj:
                    nodej[1] += vali
                    ignoreIndices.append(i)

        # Add coefficients for variables not already seen in this expression.
        for i, nodei in enumerate(newEntries):
            if i in ignoreIndices:
                continue
            entries.append(nodei)

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
            A linked list of [varIndex,value] pairs for each term in this expression.
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
                nodei[1] *= -1

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
                            "Bilinear term detected! Currently, MatrixRepresentation can only handle linear models."
                        )
                    baseResult = newEntries
                    baseConst = newConst
                else:
                    newCoef *= newConst

            baseConst *= newCoef
            if baseResult is not None:
                for nodei in baseResult:
                    nodei[1] *= newCoef

            entries = baseResult
            const = baseConst

        else:
            raise Exception(
                f'Enable to parse expression "{expr}" of type "{type(expr)}". Is it Linear?'
            )

        return entries, const

    def _ParseConstraint(self, constr: pmo.constraint):
        """
        A function to to transform a linear pyomo constraint into a

            A x <= b

        form.

        Here, A is a linked list of variable indices within the overall VAR_VEC paired with their coefficients and b is a constant.

        Parameters
        ----------
        constr: pmo.constraint
            The constraint you'd like to parse

        Returns
        -------
        A: LinkedList
            A linked list of (index,coef) pairs for the coefficients of each variable in this constraint.
        b: float
            The constant term for this expression.
        relation: Bool
            True if this constraint is an equality constraint, False if it is a "<=" inequality.
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

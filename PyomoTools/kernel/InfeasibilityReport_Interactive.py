from ..base.GenerateExpressionVisualization import GenerateExpressionVisualization
from ..util.NaturalSortKey import natural_sort_key

import pyomo.kernel as pmo
import numpy as np
import warnings
import sys
from typing import Tuple
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QTextEdit,
    QSplitter,
    QCheckBox,
    QLabel,
    QFrame,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# If there are any non-standard functions to be evaluated in the constraints, we'll define them here.
log = np.log
exp = np.exp
sin = np.sin
cos = np.cos
tan = np.tan
sqrt = np.sqrt


class InfeasibilityData:
    """
    A class to hold data for a single infeasibility.
    """

    def __init__(self, name, index, constraint, visualization):
        self.name = name
        self.index = index
        self.constraint = constraint
        self.visualization = visualization
        self.is_violated = True  # Will be set properly during analysis
        self.is_active = True  # Will be set based on constraint.is_active()
        self.violation_degree = 0.0  # Will be set during analysis

    def get_display_name(self):
        if self.index is not None:
            return f"{self.name}[{self.index}]"
        return self.name

    def get_formatted_display(self):
        """Generate the formatted display for the viewer pane."""
        var_name = self.get_display_name() + ": "
        spaces = " " * len(var_name)

        replacers = [
            lambda s: s.replace("<=", " &le;"),  # Spaced to maintain alignment
            lambda s: s.replace(">=", " &ge;"),
        ]

        def replacer(s):
            for func in replacers:
                s = func(s)
            return s

        # visualization is a string, split it into lines
        visualization_lines = self.visualization.split("\n")

        result = []
        for j in range(len(visualization_lines)):
            if j == 0:
                result.append(replacer(var_name + visualization_lines[j]))
            else:
                result.append(replacer(spaces + visualization_lines[j]))

        return result


class ContainerData:
    """
    A class to hold data for a container (list, tuple, or dict) of constraints or blocks.
    """

    def __init__(self, name, container_type):
        """
        Parameters
        ----------
        name : str
            The local name of the container.
        container_type : str
            One of 'list', 'tuple', 'dict'.
        """
        self.name = name
        self.container_type = container_type  # 'list', 'tuple', 'dict'
        self.items = {}  # index -> InfeasibilityData or BlockData
        self.num_infeasibilities = 0
        self.num_total_constraints = 0

    def add_item(self, index, item):
        """Add an item (InfeasibilityData or BlockData) to the container."""
        self.items[index] = item
        if isinstance(item, InfeasibilityData):
            if item.is_violated:
                self.num_infeasibilities += 1
            self.num_total_constraints += 1
        elif isinstance(item, BlockData):
            self.num_infeasibilities += item.num_infeasibilities
            self.num_total_constraints += item.num_total_constraints

    def get_display_name(self, show_only_infeasibilities=True):
        type_suffix = f"[{self.container_type}]"
        if show_only_infeasibilities:
            if self.num_infeasibilities > 0:
                return (
                    f"{self.name} {type_suffix} ({self.num_infeasibilities} violations)"
                )
            else:
                return f"{self.name} {type_suffix} (no violations)"
        else:
            return f"{self.name} {type_suffix} ({self.num_total_constraints} constraints, {self.num_infeasibilities} violations)"


class BlockData:
    """
    A class to hold data for a block and its constraints.
    """

    def __init__(self, name, full_name=None):
        self.name = name
        self.full_name = full_name or name
        self.constraints = []  # List of single InfeasibilityData objects
        self.constraint_containers = (
            {}
        )  # Dict of ContainerData for constraint lists/tuples/dicts
        self.sub_blocks = {}  # Dict of single child BlockData objects
        self.block_containers = {}  # Dict of ContainerData for block lists/tuples/dicts
        self.num_infeasibilities = 0
        self.num_total_constraints = 0

    def add_constraint(self, infeas_data):
        self.constraints.append(infeas_data)
        if infeas_data.is_violated:
            self.num_infeasibilities += 1
        self.num_total_constraints += 1

    def add_constraint_container(self, container_data):
        self.constraint_containers[container_data.name] = container_data
        self.num_infeasibilities += container_data.num_infeasibilities
        self.num_total_constraints += container_data.num_total_constraints

    def add_sub_block(self, block_data):
        self.sub_blocks[block_data.name] = block_data
        self.num_infeasibilities += block_data.num_infeasibilities
        self.num_total_constraints += block_data.num_total_constraints

    def add_block_container(self, container_data):
        self.block_containers[container_data.name] = container_data
        self.num_infeasibilities += container_data.num_infeasibilities
        self.num_total_constraints += container_data.num_total_constraints

    def get_display_name(self, show_only_infeasibilities=True):
        if show_only_infeasibilities:
            if self.num_infeasibilities > 0:
                return f"{self.name} ({self.num_infeasibilities} violations)"
            else:
                return f"{self.name} (no violations)"
        else:
            return f"{self.name} ({self.num_total_constraints} constraints, {self.num_infeasibilities} violations)"


class InfeasibilityReportWidget(QMainWindow):
    """
    Interactive PyQt5 widget for displaying infeasibility reports.
    """

    def __init__(
        self,
        model,
        aTol=1e-3,
        ignoreIncompleteConstraints=False,
        parent=None,
        windowTitle="Infeasibility Report",
    ):
        super().__init__(parent)
        self.model = model
        self.aTol = aTol
        self.ignoreIncompleteConstraints = ignoreIncompleteConstraints
        self.show_only_infeasibilities = True

        # Track tree state for preservation
        self.expanded_items = set()  # Paths of expanded items
        self.last_clicked_path = None  # Path of last clicked item

        # Analyze the model and build data structure
        self.root_block = self._analyze_model()

        # Setup UI
        self._setup_ui()
        self._populate_tree()

        # Set window properties
        self.setWindowTitle(windowTitle)
        self.setGeometry(100, 100, 1200, 800)

    def _setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Control panel
        control_panel = QFrame()
        control_panel.setMaximumHeight(40)  # Limit the height of the control panel
        control_panel.setContentsMargins(5, 5, 5, 5)  # Add small margins
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(5, 5, 5, 5)  # Reduce layout margins
        control_layout.setSpacing(10)  # Reduce spacing between elements

        # Filter checkbox
        self.filter_checkbox = QCheckBox("Show only violated constraints")
        self.filter_checkbox.setChecked(self.show_only_infeasibilities)
        self.filter_checkbox.stateChanged.connect(self._on_filter_changed)
        control_layout.addWidget(self.filter_checkbox)

        # Filter text box
        from PyQt5.QtWidgets import QLineEdit

        self.filter_textbox = QLineEdit()
        self.filter_textbox.setPlaceholderText("Filter by visualization text...")
        self.filter_textbox.textChanged.connect(self._on_filter_text_changed)
        control_layout.addWidget(self.filter_textbox)

        # Summary label
        self.summary_label = QLabel()
        self._update_summary_label()
        control_layout.addWidget(self.summary_label)

        control_layout.addStretch()
        main_layout.addWidget(control_panel)

        # Create splitter for two panes
        splitter = QSplitter(Qt.Horizontal)

        # Left pane - Tree view (Explorer)
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabel("Constraints by Block")
        self.tree_widget.itemClicked.connect(self._on_tree_item_clicked)
        self.tree_widget.setMaximumWidth(400)
        self.tree_widget.setMinimumWidth(250)
        self.tree_widget.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )  # Enable horizontal scrollbar
        self.tree_widget.setHorizontalScrollMode(
            QTreeWidget.ScrollPerPixel
        )  # Smooth horizontal scrolling

        # Right pane - Text viewer
        self.text_viewer = QTextEdit()
        self.text_viewer.setReadOnly(True)
        self.text_viewer.setFont(QFont("Courier", 10))
        self.text_viewer.setLineWrapMode(QTextEdit.NoWrap)  # Disable word wrapping
        self.text_viewer.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )  # Enable horizontal scrollbar
        self.text_viewer.setText("Select a constraint from the tree to view details.")

        splitter.addWidget(self.tree_widget)
        splitter.addWidget(self.text_viewer)
        splitter.setSizes([300, 900])  # Set initial sizes

        # Add splitter with stretch factor to take up remaining space
        main_layout.addWidget(
            splitter, 1
        )  # The stretch factor of 1 ensures it takes most space

    def _update_summary_label(self):
        """Update the summary label with current statistics."""
        total_violations = self.root_block.num_infeasibilities
        total_constraints = self.root_block.num_total_constraints
        self.summary_label.setText(
            f"Total: {total_constraints} constraints, {total_violations} violations"
        )

    def _analyze_model(self):
        """Analyze the pyomo model and build the data structure."""
        root_block = BlockData("Root Model")
        self._analyze_block(self.model, root_block)
        return root_block

    def _analyze_block(self, model, block_data):
        """Recursively analyze a pyomo block."""
        # Find all children from within this block
        for c in model.children():
            c_name = c.local_name if hasattr(c, "local_name") else str(c)
            full_name = c.name

            try:
                obj = getattr(model, c_name)
            except Exception:
                if ".DCC_constraint" in c_name:
                    continue
                warnings.warn(f'Warning! Could not locate child object named "{c}"')
                continue

            if isinstance(
                obj,
                (
                    pmo.variable,
                    pmo.variable_dict,
                    pmo.variable_list,
                    pmo.variable_tuple,
                    pmo.parameter,
                    pmo.parameter_dict,
                    pmo.parameter_list,
                    pmo.parameter_tuple,
                    pmo.objective,
                    pmo.objective_dict,
                    pmo.objective_list,
                    pmo.objective_tuple,
                    pmo.expression,
                    pmo.expression_dict,
                    pmo.expression_list,
                    pmo.expression_tuple,
                ),
            ):
                continue

            elif isinstance(obj, pmo.constraint_list):
                container = ContainerData(c_name, "list")
                for index in range(len(obj)):
                    infeas_data = self._process_constraint(obj[index], c_name, index)
                    container.add_item(index, infeas_data)
                block_data.add_constraint_container(container)

            elif isinstance(obj, pmo.constraint_tuple):
                container = ContainerData(c_name, "tuple")
                for index in range(len(obj)):
                    infeas_data = self._process_constraint(obj[index], c_name, index)
                    container.add_item(index, infeas_data)
                block_data.add_constraint_container(container)

            elif isinstance(obj, pmo.constraint_dict):
                container = ContainerData(c_name, "dict")
                for index in obj:
                    infeas_data = self._process_constraint(obj[index], c_name, index)
                    container.add_item(index, infeas_data)
                block_data.add_constraint_container(container)

            elif isinstance(obj, pmo.constraint):
                infeas_data = self._process_constraint(obj, c_name, None)
                block_data.add_constraint(infeas_data)

            elif isinstance(obj, pmo.block_list):
                container = ContainerData(c_name, "list")
                for index in range(len(obj)):
                    sub_block_data = BlockData(str(index), f"{full_name}[{index}]")
                    self._analyze_block(obj[index], sub_block_data)
                    container.add_item(index, sub_block_data)
                block_data.add_block_container(container)

            elif isinstance(obj, pmo.block_tuple):
                container = ContainerData(c_name, "tuple")
                for index in range(len(obj)):
                    sub_block_data = BlockData(str(index), f"{full_name}[{index}]")
                    self._analyze_block(obj[index], sub_block_data)
                    container.add_item(index, sub_block_data)
                block_data.add_block_container(container)

            elif isinstance(obj, pmo.block_dict):
                container = ContainerData(c_name, "dict")
                for index in obj:
                    sub_block_data = BlockData(str(index), f"{full_name}[{index}]")
                    self._analyze_block(obj[index], sub_block_data)
                    container.add_item(index, sub_block_data)
                block_data.add_block_container(container)

            elif isinstance(obj, pmo.block):
                sub_block_data = BlockData(c_name, full_name)
                self._analyze_block(obj, sub_block_data)
                block_data.add_sub_block(sub_block_data)

    def _process_constraint(self, constraint, name, index):
        """Process a single constraint and return InfeasibilityData."""
        # Check if constraint is active
        is_active = constraint.active

        # Generate expression strings
        try:
            visualization = GenerateExpressionVisualization(constraint.expr)
            is_feasible, violation_degree = self._test_feasibility(constraint)
            violated = not is_feasible
        except ValueError as e:
            if "value is None" in str(e):
                visualization = f"{constraint.expr}\n<Could not evaluate expression due to incomplete variable values>"
                violated = True
                violation_degree = 0.0
            else:
                raise e

        # Create infeasibility data
        infeas_data = InfeasibilityData(name, index, constraint, visualization)

        # Set feasibility and active status
        infeas_data.is_violated = violated
        infeas_data.is_active = is_active
        infeas_data.violation_degree = violation_degree

        return infeas_data

    def _test_feasibility(self, constr):
        """Test whether a constraint is feasible and return the degree of violation."""
        lower = constr.lower
        upper = constr.upper
        body = constr.body

        if body is None:
            return True, 0.0

        try:
            body_value = pmo.value(body, exception=self.ignoreIncompleteConstraints)
        except Exception:
            return self.ignoreIncompleteConstraints, 0.0

        if body_value is None:
            return self.ignoreIncompleteConstraints, 0.0

        max_violation = 0.0

        if lower is not None:
            lower_violation = lower - body_value - self.aTol
            if lower_violation > 0:
                max_violation = max(max_violation, lower_violation)

        if upper is not None:
            upper_violation = body_value - upper - self.aTol
            if upper_violation > 0:
                max_violation = max(max_violation, upper_violation)

        is_feasible = max_violation == 0
        return is_feasible, max_violation

    def _populate_tree(self):
        """Populate the tree widget with constraints."""
        # Save current tree state
        self._save_tree_state()

        self.tree_widget.clear()
        filter_text = (
            self.filter_textbox.text().strip().lower()
            if hasattr(self, "filter_textbox")
            else ""
        )
        self._add_block_to_tree(self.root_block, None, filter_text)
        self.tree_widget.collapseAll()

        # Restore tree state
        self._restore_tree_state()

    def _add_block_to_tree(self, block_data, parent_item, filter_text=""):
        """Recursively add block data to the tree."""

        # Determine if this block or any of its children match the filter
        def block_matches(block_data):
            # Check constraints
            for constraint_data in block_data.constraints:
                if self._constraint_matches_filter(constraint_data, filter_text):
                    return True
            # Check containers
            for container_data in block_data.constraint_containers.values():
                if self._container_matches_filter(container_data, filter_text):
                    return True
            # Check sub-blocks
            for sub_block_data in block_data.sub_blocks.values():
                if block_matches(sub_block_data):
                    return True
            # Check block containers
            for container_data in block_data.block_containers.values():
                if self._container_matches_filter(container_data, filter_text):
                    return True
            return False

        if filter_text and not block_matches(block_data):
            return  # Skip this block if nothing matches

        # Create tree item for this block
        if parent_item is None:
            block_item = QTreeWidgetItem(self.tree_widget)
        else:
            block_item = QTreeWidgetItem(parent_item)

        block_item.setText(
            0, block_data.get_display_name(self.show_only_infeasibilities)
        )
        block_item.setData(0, Qt.UserRole, ("block", block_data))

        # Add single constraints to this block
        for constraint_data in block_data.constraints:
            # Filter constraints if needed
            if self.show_only_infeasibilities and not constraint_data.is_violated:
                continue
            if filter_text and not self._constraint_matches_filter(
                constraint_data, filter_text
            ):
                continue

            constraint_item = QTreeWidgetItem(block_item)
            constraint_item.setText(0, constraint_data.get_display_name())
            constraint_item.setData(0, Qt.UserRole, ("constraint", constraint_data))

            # Color code violated constraints
            if constraint_data.is_violated:
                constraint_item.setForeground(0, Qt.red)

            # Grey out inactive constraints
            if not constraint_data.is_active:
                constraint_item.setForeground(0, Qt.gray)

        # Add constraint containers (lists, tuples, dicts)
        for container_data in block_data.constraint_containers.values():
            # Skip empty containers if filtering
            if (
                self.show_only_infeasibilities
                and container_data.num_infeasibilities == 0
            ):
                continue
            if filter_text and not self._container_matches_filter(
                container_data, filter_text
            ):
                continue
            self._add_constraint_container_to_tree(
                container_data, block_item, filter_text
            )

        # Add single sub-blocks
        for sub_block_data in block_data.sub_blocks.values():
            # Skip empty blocks if filtering
            if (
                self.show_only_infeasibilities
                and sub_block_data.num_infeasibilities == 0
            ):
                continue
            self._add_block_to_tree(sub_block_data, block_item, filter_text)

        # Add block containers (lists, tuples, dicts)
        for container_data in block_data.block_containers.values():
            # Skip empty containers if filtering
            if (
                self.show_only_infeasibilities
                and container_data.num_infeasibilities == 0
            ):
                continue
            if filter_text and not self._container_matches_filter(
                container_data, filter_text
            ):
                continue
            self._add_block_container_to_tree(container_data, block_item, filter_text)

    def _add_constraint_container_to_tree(
        self, container_data, parent_item, filter_text=""
    ):
        """Add a constraint container (list, tuple, dict) to the tree."""
        if filter_text and not self._container_matches_filter(
            container_data, filter_text
        ):
            return
        container_item = QTreeWidgetItem(parent_item)
        container_item.setText(
            0, container_data.get_display_name(self.show_only_infeasibilities)
        )
        container_item.setData(0, Qt.UserRole, ("constraint_container", container_data))

        # Add each indexed constraint (sorted with natural numeric ordering)
        sorted_indices = sorted(container_data.items.keys(), key=natural_sort_key)
        for index in sorted_indices:
            constraint_data = container_data.items[index]
            # Filter constraints if needed
            if self.show_only_infeasibilities and not constraint_data.is_violated:
                continue
            if filter_text and not self._constraint_matches_filter(
                constraint_data, filter_text
            ):
                continue

            constraint_item = QTreeWidgetItem(container_item)
            constraint_item.setText(0, f"[{index}]")
            constraint_item.setData(0, Qt.UserRole, ("constraint", constraint_data))

            # Color code violated constraints
            if constraint_data.is_violated:
                constraint_item.setForeground(0, Qt.red)

            # Grey out inactive constraints
            if not constraint_data.is_active:
                constraint_item.setForeground(0, Qt.gray)

    def _add_block_container_to_tree(self, container_data, parent_item, filter_text=""):
        """Add a block container (list, tuple, dict) to the tree."""
        if filter_text and not self._container_matches_filter(
            container_data, filter_text
        ):
            return
        container_item = QTreeWidgetItem(parent_item)
        container_item.setText(
            0, container_data.get_display_name(self.show_only_infeasibilities)
        )
        container_item.setData(0, Qt.UserRole, ("block_container", container_data))

        # Add each indexed block (sorted with natural numeric ordering)
        sorted_indices = sorted(container_data.items.keys(), key=natural_sort_key)
        for index in sorted_indices:
            sub_block_data = container_data.items[index]
            # Skip empty blocks if filtering
            if (
                self.show_only_infeasibilities
                and sub_block_data.num_infeasibilities == 0
            ):
                continue
            if filter_text and not self._block_matches_filter(
                sub_block_data, filter_text
            ):
                continue

            # Create an intermediate item for the index
            index_item = QTreeWidgetItem(container_item)
            index_item.setText(
                0,
                f"[{index}] "
                + sub_block_data.get_display_name(self.show_only_infeasibilities),
            )
            index_item.setData(0, Qt.UserRole, ("block", sub_block_data))

            # Add the block's contents directly under the index item
            # Add single constraints
            for constraint_data in sub_block_data.constraints:
                if self.show_only_infeasibilities and not constraint_data.is_violated:
                    continue
                if filter_text and not self._constraint_matches_filter(
                    constraint_data, filter_text
                ):
                    continue
                constraint_item = QTreeWidgetItem(index_item)
                constraint_item.setText(0, constraint_data.get_display_name())
                constraint_item.setData(0, Qt.UserRole, ("constraint", constraint_data))
                if constraint_data.is_violated:
                    constraint_item.setForeground(0, Qt.red)
                if not constraint_data.is_active:
                    constraint_item.setForeground(0, Qt.gray)

            # Add constraint containers
            for child_container in sub_block_data.constraint_containers.values():
                if (
                    self.show_only_infeasibilities
                    and child_container.num_infeasibilities == 0
                ):
                    continue
                if filter_text and not self._container_matches_filter(
                    child_container, filter_text
                ):
                    continue
                self._add_constraint_container_to_tree(
                    child_container, index_item, filter_text
                )

            # Add single sub-blocks
            for child_block in sub_block_data.sub_blocks.values():
                if (
                    self.show_only_infeasibilities
                    and child_block.num_infeasibilities == 0
                ):
                    continue
                self._add_block_to_tree(child_block, index_item, filter_text)

            # Add block containers
            for child_container in sub_block_data.block_containers.values():
                if (
                    self.show_only_infeasibilities
                    and child_container.num_infeasibilities == 0
                ):
                    continue
                if filter_text and not self._container_matches_filter(
                    child_container, filter_text
                ):
                    continue
                self._add_block_container_to_tree(
                    child_container, index_item, filter_text
                )

    def _constraint_matches_filter(self, constraint_data, filter_text):
        if not filter_text:
            return True
        # Check if filter_text is in the visualization (case-insensitive)
        return filter_text in constraint_data.visualization.lower()

    def _container_matches_filter(self, container_data, filter_text):
        if not filter_text:
            return True
        for item in container_data.items.values():
            if isinstance(item, InfeasibilityData):
                if self._constraint_matches_filter(item, filter_text):
                    return True
            elif isinstance(item, BlockData):
                if self._block_matches_filter(item, filter_text):
                    return True
        return False

    def _block_matches_filter(self, block_data, filter_text):
        if not filter_text:
            return True
        # Check constraints
        for constraint_data in block_data.constraints:
            if self._constraint_matches_filter(constraint_data, filter_text):
                return True
        # Check containers
        for container_data in block_data.constraint_containers.values():
            if self._container_matches_filter(container_data, filter_text):
                return True
        # Check sub-blocks
        for sub_block_data in block_data.sub_blocks.values():
            if self._block_matches_filter(sub_block_data, filter_text):
                return True
        # Check block containers
        for container_data in block_data.block_containers.values():
            if self._container_matches_filter(container_data, filter_text):
                return True
        return False

    def _on_filter_text_changed(self, text):
        self._populate_tree()

    def _on_filter_changed(self, state):
        """Handle filter checkbox state change."""
        self.show_only_infeasibilities = state == Qt.Checked
        self._populate_tree()
        self.text_viewer.setText("Select a constraint from the tree to view details.")

    def _on_tree_item_clicked(self, item, column):
        """Handle tree item click."""
        data = item.data(0, Qt.UserRole)
        if data is None:
            return

        # Save the path of the clicked item
        self.last_clicked_path = self._get_item_path(item)

        item_type, item_data = data

        if item_type == "constraint":
            # Display constraint details
            self._display_constraint_details(item_data)
        elif item_type == "block":
            # Display block summary
            self._display_block_summary(item_data)
        elif item_type in ("constraint_container", "block_container"):
            # Display container summary
            self._display_container_summary(item_data)

    def _display_constraint_details(self, constraint_data):
        """Display detailed information about a constraint."""
        lines = constraint_data.get_formatted_display()

        # Create formatted text
        text = "<br>".join(lines)

        # Add violation status
        if constraint_data.is_violated:
            status = f"VIOLATED (degree of violation: {constraint_data.violation_degree:.6e})"
        else:
            status = "SATISFIED"
        if not constraint_data.is_active:
            status += " (Inactive)"

        color = "red" if constraint_data.is_violated else "green"
        if not constraint_data.is_active:
            color = "gray"

        formatted_text = f"""<h3 style="color: {color};">Constraint Status: {status}</h3>
<pre style="font-family: 'Courier New', monospace; font-size: 10pt;">
{text}
</pre>"""

        self.text_viewer.setHtml(formatted_text)

    def _display_block_summary(self, block_data):
        """Display summary information about a block."""
        total_constraints = len(block_data.constraints)
        violated_constraints = sum(1 for c in block_data.constraints if c.is_violated)

        summary = f"""<h3>Block: {block_data.name}</h3>
<p><strong>Full Name:</strong> {block_data.full_name}</p>
<p><strong>Direct Constraints:</strong> {total_constraints}</p>
<p><strong>Violated Constraints:</strong> {violated_constraints}</p>
<p><strong>Sub-blocks:</strong> {len(block_data.sub_blocks)}</p>
<p><strong>Constraint Containers:</strong> {len(block_data.constraint_containers)}</p>
<p><strong>Block Containers:</strong> {len(block_data.block_containers)}</p>
<p><strong>Total Constraints (including sub-blocks):</strong> {block_data.num_total_constraints}</p>
<p><strong>Total Violations (including sub-blocks):</strong> {block_data.num_infeasibilities}</p>
"""

        if violated_constraints > 0:
            summary += "<h4>Violated Constraints in this block:</h4><ul>"
            for constraint_data in block_data.constraints:
                if constraint_data.is_violated:
                    summary += f"<li style='color: red;'>{constraint_data.get_display_name()}</li>"
            summary += "</ul>"

        self.text_viewer.setHtml(summary)

    def _display_container_summary(self, container_data):
        """Display summary information about a container."""
        total_items = len(container_data.items)

        # Check if this is a constraint container or block container
        first_item = (
            next(iter(container_data.items.values()), None)
            if container_data.items
            else None
        )
        is_constraint_container = (
            isinstance(first_item, InfeasibilityData) if first_item else True
        )

        if is_constraint_container:
            violated_items = sum(
                1 for c in container_data.items.values() if c.is_violated
            )
            summary = f"""<h3>Constraint Container: {container_data.name}</h3>
<p><strong>Container Type:</strong> {container_data.container_type}</p>
<p><strong>Number of Constraints:</strong> {total_items}</p>
<p><strong>Violated Constraints:</strong> {violated_items}</p>
"""
            if violated_items > 0:
                summary += "<h4>Violated Constraints:</h4><ul>"
                for index, constraint_data in container_data.items.items():
                    if constraint_data.is_violated:
                        summary += f"<li style='color: red;'>[{index}]</li>"
                summary += "</ul>"
        else:
            summary = f"""<h3>Block Container: {container_data.name}</h3>
<p><strong>Container Type:</strong> {container_data.container_type}</p>
<p><strong>Number of Blocks:</strong> {total_items}</p>
<p><strong>Total Constraints:</strong> {container_data.num_total_constraints}</p>
<p><strong>Total Violations:</strong> {container_data.num_infeasibilities}</p>
"""
            if container_data.num_infeasibilities > 0:
                summary += "<h4>Blocks with Violations:</h4><ul>"
                for index, block_data in container_data.items.items():
                    if block_data.num_infeasibilities > 0:
                        summary += f"<li style='color: red;'>[{index}] ({block_data.num_infeasibilities} violations)</li>"
                summary += "</ul>"

        self.text_viewer.setHtml(summary)

    def _get_item_path(self, item):
        """Get the full path of a tree item as a tuple of text labels."""
        path = []
        current = item
        while current is not None:
            path.insert(0, current.text(0))
            current = current.parent()
        return tuple(path)

    def _save_tree_state(self):
        """Save the expanded state of all tree items."""
        self.expanded_items = set()
        iterator = QTreeWidgetItemIterator(self.tree_widget)
        while iterator.value():
            item = iterator.value()
            if item.isExpanded():
                path = self._get_item_path(item)
                self.expanded_items.add(path)
            iterator += 1

    def _restore_tree_state(self):
        """Restore the expanded state of tree items."""
        # First, restore expanded states
        iterator = QTreeWidgetItemIterator(self.tree_widget)
        last_clicked_item = None

        while iterator.value():
            item = iterator.value()
            path = self._get_item_path(item)

            # Restore expanded state
            if path in self.expanded_items:
                item.setExpanded(True)

            # Track if this is the last clicked item
            if path == self.last_clicked_path:
                last_clicked_item = item

            iterator += 1

        # Scroll to and select the last clicked item if found
        if last_clicked_item is not None:
            self.tree_widget.scrollToItem(last_clicked_item)
            self.tree_widget.setCurrentItem(last_clicked_item)


class InfeasibilityReport_Interactive:
    """
    Interactive version of InfeasibilityReport using PyQt5.

    This class creates and manages the interactive window for displaying
    infeasibility reports with an explorer pane and viewer pane.
    """

    def __init__(self, model, aTol=1e-3, ignoreIncompleteConstraints=False):
        """
        Constructor for interactive infeasibility report.

        Parameters
        ----------
        model: pmo.block
            The pyomo model (containing a solution) to analyze.
        aTol: float (optional, Default = 1e-3)
            The absolute tolerance for determining constraint violations.
        ignoreIncompleteConstraints: bool (optional, Default = False)
            Whether to ignore constraints with incomplete variable values.
        """
        self.model = model
        self.aTol = aTol
        self.ignoreIncompleteConstraints = ignoreIncompleteConstraints
        self.app = None
        self.widget = None

    def show(
        self,
        windowTitle="Infeasibility Report",
        geometry: Tuple[int, int, int, int] = None,
    ):
        """
        Display the interactive infeasibility report window.
        """
        # Create QApplication if it doesn't exist
        if QApplication.instance() is None:
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()

        # Create and show the widget
        self.widget = InfeasibilityReportWidget(
            self.model,
            self.aTol,
            self.ignoreIncompleteConstraints,
            windowTitle=windowTitle,
        )
        if geometry is not None:
            self.widget.setGeometry(*geometry)
        self.widget.show()

        # If we created the app, run the event loop
        if self.app and not hasattr(self.app, "_running"):
            self.app._running = True
            self.app.exec_()

    def get_widget(self):
        """
        Get the QWidget for embedding in other applications.

        Returns
        -------
        InfeasibilityReportWidget
            The widget that can be embedded in other Qt applications.
        """
        if self.widget is None:
            self.widget = InfeasibilityReportWidget(
                self.model, self.aTol, self.ignoreIncompleteConstraints
            )
        return self.widget


def create_infeasibility_report_interactive(
    model, aTol=1e-3, ignoreIncompleteConstraints=False
):
    """
    Convenience function to create and show an interactive infeasibility report.

    Parameters
    ----------
    model: pmo.block
        The pyomo model (containing a solution) to analyze.
    aTol: float (optional, Default = 1e-3)
        The absolute tolerance for determining constraint violations.
    ignoreIncompleteConstraints: bool (optional, Default = False)
        Whether to ignore constraints with incomplete variable values.

    Returns
    -------
    InfeasibilityReport_Interactive
        The interactive report object.
    """
    report = InfeasibilityReport_Interactive(model, aTol, ignoreIncompleteConstraints)
    report.show()
    return report


# Example usage
if __name__ == "__main__":
    # This would be used with an actual pyomo model
    print("Interactive Infeasibility Report")
    print("Usage:")
    print(
        "  from PyomoTools.kernel.InfeasibilityReport_Interactive import create_infeasibility_report_interactive"
    )
    print("  report = create_infeasibility_report_interactive(your_model)")

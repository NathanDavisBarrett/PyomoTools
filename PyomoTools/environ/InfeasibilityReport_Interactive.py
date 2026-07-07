from ..base.GenerateExpressionVisualization import GenerateExpressionVisualization
from ..util.NaturalSortKey import natural_sort_key
import html

import pyomo.environ as pyo
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

    def __init__(self, name, index, constraint, visualization=None):
        self.name = name
        self.index = index
        self.constraint = constraint
        self.visualization = visualization
        self.is_violated = True  # Will be set properly during analysis
        self.is_active = True  # Will be set properly during analysis
        self.violation_degree = 0.0  # Will be set during analysis

    def get_visualization(self):
        if self.visualization is None:
            try:
                self.visualization = GenerateExpressionVisualization(
                    self.constraint.expr
                )
            except ValueError as e:
                if "value is None" in str(e) or "error" in str(e).lower():
                    self.visualization = f"{self.constraint.expr}\n<Could not evaluate expression due to incomplete variable values>"
                else:
                    raise e
            except Exception as e:
                self.visualization = (
                    f"{self.constraint.expr}\n<Error evaluating constraint: {str(e)}>"
                )
        return self.visualization

    def get_display_name(self):
        if self.index is not None:
            return f"{self.name}[{self.index}]"
        return self.name

    def get_formatted_display(self):
        """Generate the formatted display for the viewer pane."""
        var_name = self.get_display_name() + ": "
        spaces = " " * len(var_name)

        replacers = [
            lambda s: html.escape(s),  # Protect HTML special characters
            lambda s: s.replace("<=", " &le;"),  # Spaced to maintain alignment
            lambda s: s.replace(">=", " &ge;"),
        ]

        def replacer(s):
            for func in replacers:
                s = func(s)
            return s

        visualization_lines = self.get_visualization().split("\n")

        result = []
        for j in range(len(visualization_lines)):
            if j == 0:
                result.append(replacer(var_name + visualization_lines[j]))
            else:
                result.append(replacer(spaces + visualization_lines[j]))

        return result


class ContainerData:
    """
    A class to hold data for a container of constraints.
    """

    def __init__(self, name, container_type):
        """
        Parameters
        ----------
        name : str
            The local name of the container.
        container_type : str
            One of 'dict', etc.
        """
        self.name = name
        self.container_type = container_type
        self.items = {}  # index -> InfeasibilityData
        self.num_infeasibilities = 0
        self.num_total_constraints = 0

    def add_item(self, index, item):
        self.items[index] = item
        if isinstance(item, InfeasibilityData):
            if item.is_violated:
                self.num_infeasibilities += 1
            self.num_total_constraints += 1

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
    (In environ, this is just the root model since sub-blocks are not supported).
    """

    def __init__(self, name, full_name=None):
        self.name = name
        self.full_name = full_name or name
        self.constraints = []  # List of single InfeasibilityData objects
        self.constraint_containers = {}  # Dict of ContainerData for indexed constraints
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

        self.expanded_items = set()
        self.last_clicked_path = None

        self.root_block = self._analyze_model()

        self._setup_ui()
        self._populate_tree()

        self.setWindowTitle(windowTitle)
        self.setGeometry(100, 100, 1200, 800)

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        control_panel = QFrame()
        control_panel.setMaximumHeight(40)
        control_panel.setContentsMargins(5, 5, 5, 5)
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(5, 5, 5, 5)
        control_layout.setSpacing(10)

        self.filter_checkbox = QCheckBox("Show only violated constraints")
        self.filter_checkbox.setChecked(self.show_only_infeasibilities)
        self.filter_checkbox.stateChanged.connect(self._on_filter_changed)
        control_layout.addWidget(self.filter_checkbox)

        from PyQt5.QtWidgets import QLineEdit

        self.filter_textbox = QLineEdit()
        self.filter_textbox.setPlaceholderText("Filter by expression text...")
        self.filter_textbox.textChanged.connect(self._on_filter_text_changed)
        control_layout.addWidget(self.filter_textbox)

        self.summary_label = QLabel()
        self._update_summary_label()
        control_layout.addWidget(self.summary_label)

        control_layout.addStretch()
        main_layout.addWidget(control_panel)

        splitter = QSplitter(Qt.Horizontal)

        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabel("Constraints by Block")
        self.tree_widget.itemClicked.connect(self._on_tree_item_clicked)
        self.tree_widget.setMaximumWidth(400)
        self.tree_widget.setMinimumWidth(250)
        self.tree_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tree_widget.setHorizontalScrollMode(QTreeWidget.ScrollPerPixel)

        self.text_viewer = QTextEdit()
        self.text_viewer.setReadOnly(True)
        self.text_viewer.setFont(QFont("Courier", 10))
        self.text_viewer.setLineWrapMode(QTextEdit.NoWrap)
        self.text_viewer.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.text_viewer.setText("Select a constraint from the tree to view details.")

        splitter.addWidget(self.tree_widget)
        splitter.addWidget(self.text_viewer)
        splitter.setSizes([300, 900])

        main_layout.addWidget(splitter, 1)

    def _update_summary_label(self):
        total_violations = self.root_block.num_infeasibilities
        total_constraints = self.root_block.num_total_constraints
        self.summary_label.setText(
            f"Total: {total_constraints} constraints, {total_violations} violations"
        )

    def _analyze_model(self):
        root_block = BlockData(self.model.name or "Root Model")

        # In environ, we just extract all Constraints at the model level
        for comp in self.model.component_objects(ctype=pyo.Constraint, active=True):
            c_name = comp.local_name

            # Scalar constraint
            if not comp.is_indexed():
                infeas_data = self._process_constraint(comp, c_name, None)
                root_block.add_constraint(infeas_data)
            # Indexed constraint
            else:
                container_type = "indexed"
                container = ContainerData(c_name, container_type)
                for index in comp:
                    constr_data = comp[index]
                    infeas_data = self._process_constraint(constr_data, c_name, index)
                    container.add_item(index, infeas_data)
                root_block.add_constraint_container(container)

        return root_block

    def _process_constraint(self, constraint, name, index):
        is_active = constraint.active

        is_feasible, violation_degree = self._test_feasibility(constraint)
        violated = not is_feasible

        infeas_data = InfeasibilityData(name, index, constraint, None)
        infeas_data.is_violated = violated
        infeas_data.is_active = is_active
        infeas_data.violation_degree = violation_degree

        return infeas_data

    def _test_feasibility(self, constr):
        lower = constr.lower
        upper = constr.upper
        body = constr.body

        if body is None:
            return True, 0.0

        try:
            # using pyo.value which respects exception flag nicely for environ
            body_value = pyo.value(body, exception=not self.ignoreIncompleteConstraints)
        except Exception:
            return self.ignoreIncompleteConstraints, 0.0

        if body_value is None:
            return self.ignoreIncompleteConstraints, 0.0

        max_violation = 0.0

        try:
            if lower is not None:
                lower_val = pyo.value(lower, exception=False)
                if lower_val is not None:
                    lower_violation = lower_val - body_value - self.aTol
                    if lower_violation > 0:
                        max_violation = max(max_violation, lower_violation)

            if upper is not None:
                upper_val = pyo.value(upper, exception=False)
                if upper_val is not None:
                    upper_violation = body_value - upper_val - self.aTol
                    if upper_violation > 0:
                        max_violation = max(max_violation, upper_violation)
        except Exception:
            return self.ignoreIncompleteConstraints, 0.0

        is_feasible = max_violation == 0
        return is_feasible, max_violation

    def _populate_tree(self):
        self._save_tree_state()
        self.tree_widget.clear()
        filter_text = (
            self.filter_textbox.text().strip().lower()
            if hasattr(self, "filter_textbox")
            else ""
        )
        self._add_block_to_tree(self.root_block, None, filter_text)
        self.tree_widget.collapseAll()
        self._restore_tree_state()

    def _add_block_to_tree(self, block_data, parent_item, filter_text=""):
        def block_matches(bd):
            for cd in bd.constraints:
                if self._constraint_matches_filter(cd, filter_text):
                    return True
            for cont_d in bd.constraint_containers.values():
                if self._container_matches_filter(cont_d, filter_text):
                    return True
            return False

        if filter_text and not block_matches(block_data):
            return

        if parent_item is None:
            block_item = QTreeWidgetItem(self.tree_widget)
        else:
            block_item = QTreeWidgetItem(parent_item)

        block_item.setText(
            0, block_data.get_display_name(self.show_only_infeasibilities)
        )
        block_item.setData(0, Qt.UserRole, ("block", block_data))

        for constraint_data in block_data.constraints:
            if self.show_only_infeasibilities and not constraint_data.is_violated:
                continue
            if filter_text and not self._constraint_matches_filter(
                constraint_data, filter_text
            ):
                continue
            constraint_item = QTreeWidgetItem(block_item)
            constraint_item.setText(0, constraint_data.get_display_name())
            constraint_item.setData(0, Qt.UserRole, ("constraint", constraint_data))
            if constraint_data.is_violated:
                constraint_item.setForeground(0, Qt.red)
            if not constraint_data.is_active:
                constraint_item.setForeground(0, Qt.gray)

        for container_data in block_data.constraint_containers.values():
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

    def _add_constraint_container_to_tree(
        self, container_data, parent_item, filter_text=""
    ):
        if filter_text and not self._container_matches_filter(
            container_data, filter_text
        ):
            return
        container_item = QTreeWidgetItem(parent_item)
        container_item.setText(
            0, container_data.get_display_name(self.show_only_infeasibilities)
        )
        container_item.setData(0, Qt.UserRole, ("constraint_container", container_data))

        sorted_indices = sorted(container_data.items.keys(), key=natural_sort_key)
        for index in sorted_indices:
            constraint_data = container_data.items[index]
            if self.show_only_infeasibilities and not constraint_data.is_violated:
                continue
            if filter_text and not self._constraint_matches_filter(
                constraint_data, filter_text
            ):
                continue

            constraint_item = QTreeWidgetItem(container_item)
            constraint_item.setText(0, f"[{index}]")
            constraint_item.setData(0, Qt.UserRole, ("constraint", constraint_data))

            if constraint_data.is_violated:
                constraint_item.setForeground(0, Qt.red)
            if not constraint_data.is_active:
                constraint_item.setForeground(0, Qt.gray)

    def _constraint_matches_filter(self, constraint_data, filter_text):
        if not filter_text:
            return True
        return filter_text in constraint_data.visualization.lower()

    def _container_matches_filter(self, container_data, filter_text):
        if not filter_text:
            return True
        for item in container_data.items.values():
            if isinstance(item, InfeasibilityData):
                if self._constraint_matches_filter(item, filter_text):
                    return True
        return False

    def _on_filter_text_changed(self, text):
        self._populate_tree()

    def _on_filter_changed(self, state):
        self.show_only_infeasibilities = state == Qt.Checked
        self._populate_tree()
        self.text_viewer.setText("Select a constraint from the tree to view details.")

    def _on_tree_item_clicked(self, item, column):
        data = item.data(0, Qt.UserRole)
        if data is None:
            return
        self.last_clicked_path = self._get_item_path(item)
        item_type, item_data = data
        if item_type == "constraint":
            self._display_constraint_details(item_data)
        elif item_type == "block":
            self._display_block_summary(item_data)
        elif item_type in ("constraint_container", "block_container"):
            self._display_container_summary(item_data)

    def _display_constraint_details(self, constraint_data):
        lines = constraint_data.get_formatted_display()
        text = "<br>".join(lines)

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
        total_constraints = len(block_data.constraints)
        violated_constraints = sum(1 for c in block_data.constraints if c.is_violated)

        summary = f"""<h3>Model: {block_data.name}</h3>
<p><strong>Direct Constraints:</strong> {total_constraints}</p>
<p><strong>Violated Direct Constraints:</strong> {violated_constraints}</p>
<p><strong>Constraint Containers:</strong> {len(block_data.constraint_containers)}</p>
<p><strong>Total Constraints:</strong> {block_data.num_total_constraints}</p>
<p><strong>Total Violations:</strong> {block_data.num_infeasibilities}</p>
"""
        if violated_constraints > 0:
            summary += "<h4>Violated Constraints:</h4><ul>"
            for constraint_data in block_data.constraints:
                if constraint_data.is_violated:
                    summary += f"<li style='color: red;'>{constraint_data.get_display_name()}</li>"
            summary += "</ul>"

        self.text_viewer.setHtml(summary)

    def _display_container_summary(self, container_data):
        total_items = len(container_data.items)
        violated_items = sum(1 for c in container_data.items.values() if c.is_violated)
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

        self.text_viewer.setHtml(summary)

    def _get_item_path(self, item):
        path = []
        current = item
        while current is not None:
            path.insert(0, current.text(0))
            current = current.parent()
        return tuple(path)

    def _save_tree_state(self):
        self.expanded_items = set()
        iterator = QTreeWidgetItemIterator(self.tree_widget)
        while iterator.value():
            item = iterator.value()
            if item.isExpanded():
                self.expanded_items.add(self._get_item_path(item))
            iterator += 1

    def _restore_tree_state(self):
        iterator = QTreeWidgetItemIterator(self.tree_widget)
        last_clicked_item = None
        while iterator.value():
            item = iterator.value()
            path = self._get_item_path(item)
            if path in self.expanded_items:
                item.setExpanded(True)
            if path == self.last_clicked_path:
                last_clicked_item = item
            iterator += 1
        if last_clicked_item is not None:
            self.tree_widget.scrollToItem(last_clicked_item)
            self.tree_widget.setCurrentItem(last_clicked_item)


class InfeasibilityReport_Interactive:
    """
    Interactive version of InfeasibilityReport using PyQt5.
    """

    def __init__(self, model, aTol=1e-3, ignoreIncompleteConstraints=False):
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
        if QApplication.instance() is None:
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()

        self.widget = InfeasibilityReportWidget(
            self.model,
            self.aTol,
            self.ignoreIncompleteConstraints,
            windowTitle=windowTitle,
        )
        if geometry is not None:
            self.widget.setGeometry(*geometry)
        self.widget.show()

        if self.app and not hasattr(self.app, "_running"):
            self.app._running = True
            self.app.exec_()

    def get_widget(self):
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
    """
    return InfeasibilityReport_Interactive(model, aTol, ignoreIncompleteConstraints)

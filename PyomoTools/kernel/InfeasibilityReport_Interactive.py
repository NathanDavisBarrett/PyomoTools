from ..base.GenerateExpressionString import GenerateExpressionStrings

import pyomo.kernel as pmo
import re
import numpy as np
import warnings
import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QTreeWidget,
    QTreeWidgetItem,
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

    def __init__(self, name, index, constraint, expr_str, substituted_expr_str):
        self.name = name
        self.index = index
        self.constraint = constraint
        self.expr_str = expr_str
        self.substituted_expr_str = substituted_expr_str
        self.is_violated = True  # Will be set properly during analysis

    def get_display_name(self):
        if self.index is not None:
            return f"{self.name}[{self.index}]"
        return self.name

    def get_formatted_display(self):
        """Generate the 4-line formatted display for the viewer pane."""
        var_name = self.get_display_name() + ":"
        spaces = " " * (len(var_name) + 1)

        # Create shortened string with excess whitespace removed
        shortened_str = re.sub(" +", " ", self.substituted_expr_str)

        # Find the divider and evaluate
        dividers = ["==", "<=", ">="]
        divider = None
        for d in dividers:
            if d in shortened_str:
                divider = d
                break

        if divider is None:
            eval_str = "N/A"
        else:
            try:
                div_index = shortened_str.index(divider)
                lhs = shortened_str[:div_index].strip()
                rhs = shortened_str[div_index + 2 :].strip()
                lhs_val = eval(lhs)
                rhs_val = eval(rhs)
                eval_str = f"{lhs_val} {divider} {rhs_val}"
            except Exception:
                eval_str = "Evaluation Error"

        return [
            f"{var_name} {self.expr_str}",
            f"{spaces}{self.substituted_expr_str}",
            f"{spaces}{shortened_str}",
            f"{spaces}{eval_str}",
        ]


class BlockData:
    """
    A class to hold data for a block and its constraints.
    """

    def __init__(self, name, full_name=None):
        self.name = name
        self.full_name = full_name or name
        self.constraints = []  # List of InfeasibilityData objects
        self.sub_blocks = {}  # Dict of child BlockData objects
        self.num_infeasibilities = 0
        self.num_total_constraints = 0

    def add_constraint(self, infeas_data):
        self.constraints.append(infeas_data)
        if infeas_data.is_violated:
            self.num_infeasibilities += 1
        self.num_total_constraints += 1

    def add_sub_block(self, block_data):
        self.sub_blocks[block_data.name] = block_data
        self.num_infeasibilities += block_data.num_infeasibilities
        self.num_total_constraints += block_data.num_total_constraints

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
        self, model, aTol=1e-3, ignoreIncompleteConstraints=False, parent=None
    ):
        super().__init__(parent)
        self.model = model
        self.aTol = aTol
        self.ignoreIncompleteConstraints = ignoreIncompleteConstraints
        self.show_only_infeasibilities = True

        # Analyze the model and build data structure
        self.root_block = self._analyze_model()

        # Setup UI
        self._setup_ui()
        self._populate_tree()

        # Set window properties
        self.setWindowTitle("Infeasibility Report")
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

            elif isinstance(obj, (pmo.constraint_list, pmo.constraint_tuple)):
                for index in range(len(obj)):
                    self._process_constraint(obj[index], str(c), index, block_data)

            elif isinstance(obj, pmo.constraint_dict):
                for index in obj:
                    self._process_constraint(obj[index], str(c), index, block_data)

            elif isinstance(obj, pmo.constraint):
                self._process_constraint(obj, c_name, None, block_data)

            elif isinstance(obj, (pmo.block_list, pmo.block_tuple)):
                for index in range(len(obj)):
                    sub_name = f"{c_name}[{index}]"
                    sub_block_data = BlockData(sub_name, f"{full_name}[{index}]")
                    self._analyze_block(obj[index], sub_block_data)
                    block_data.add_sub_block(sub_block_data)

            elif isinstance(obj, pmo.block_dict):
                for index in obj:
                    sub_name = f"{c_name}[{index}]"
                    sub_block_data = BlockData(sub_name, f"{full_name}[{index}]")
                    self._analyze_block(obj[index], sub_block_data)
                    block_data.add_sub_block(sub_block_data)

            elif isinstance(obj, pmo.block):
                sub_block_data = BlockData(c_name, full_name)
                self._analyze_block(obj, sub_block_data)
                block_data.add_sub_block(sub_block_data)

    def _process_constraint(self, constraint, name, index, block_data):
        """Process a single constraint and add it to the block data."""
        # Generate expression strings
        expr_str, substituted_expr_str = GenerateExpressionStrings(constraint.expr)

        # Create infeasibility data
        infeas_data = InfeasibilityData(
            name, index, constraint, expr_str, substituted_expr_str
        )

        # Test feasibility
        infeas_data.is_violated = not self._test_feasibility(constraint)

        # Add to block
        block_data.add_constraint(infeas_data)

    def _test_feasibility(self, constr):
        """Test whether a constraint is feasible."""
        lower = constr.lower
        upper = constr.upper
        body = constr.body

        if body is None:
            return True

        try:
            body_value = pmo.value(body, exception=self.ignoreIncompleteConstraints)
        except Exception:
            return self.ignoreIncompleteConstraints

        if body_value is None:
            return self.ignoreIncompleteConstraints

        if lower is not None:
            if body_value < lower - self.aTol:
                return False

        if upper is not None:
            if body_value > upper + self.aTol:
                return False

        return True

    def _populate_tree(self):
        """Populate the tree widget with constraints."""
        self.tree_widget.clear()
        self._add_block_to_tree(self.root_block, None)
        self.tree_widget.expandAll()

    def _add_block_to_tree(self, block_data, parent_item):
        """Recursively add block data to the tree."""
        # Create tree item for this block
        if parent_item is None:
            block_item = QTreeWidgetItem(self.tree_widget)
        else:
            block_item = QTreeWidgetItem(parent_item)

        block_item.setText(
            0, block_data.get_display_name(self.show_only_infeasibilities)
        )
        block_item.setData(0, Qt.UserRole, ("block", block_data))

        # Add constraints to this block
        for constraint_data in block_data.constraints:
            # Filter constraints if needed
            if self.show_only_infeasibilities and not constraint_data.is_violated:
                continue

            constraint_item = QTreeWidgetItem(block_item)
            constraint_item.setText(0, constraint_data.get_display_name())
            constraint_item.setData(0, Qt.UserRole, ("constraint", constraint_data))

            # Color code violated constraints
            if constraint_data.is_violated:
                constraint_item.setForeground(0, Qt.red)

        # Add sub-blocks
        for sub_block_data in block_data.sub_blocks.values():
            # Skip empty blocks if filtering
            if (
                self.show_only_infeasibilities
                and sub_block_data.num_infeasibilities == 0
            ):
                continue
            self._add_block_to_tree(sub_block_data, block_item)

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

        item_type, item_data = data

        if item_type == "constraint":
            # Display constraint details
            self._display_constraint_details(item_data)
        elif item_type == "block":
            # Display block summary
            self._display_block_summary(item_data)

    def _display_constraint_details(self, constraint_data):
        """Display detailed information about a constraint."""
        lines = constraint_data.get_formatted_display()

        # Create formatted text
        text = "\n".join(lines)

        # Add violation status
        status = "VIOLATED" if constraint_data.is_violated else "SATISFIED"
        color = "red" if constraint_data.is_violated else "green"

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

    def show(self):
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
            self.model, self.aTol, self.ignoreIncompleteConstraints
        )
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

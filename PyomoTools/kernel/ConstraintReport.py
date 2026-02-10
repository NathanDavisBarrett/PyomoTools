import pyomo.kernel as pmo
import numpy as np
import warnings


class ConstraintReport:
    """
    A class to display all the constraints in a model

    Constructor Parameters
    ----------------------
    model: pmo.block
        The pyomo model (containing a solution) that you'd like to generate the infeasibility report for.
    """

    def __init__(
        self,
        model: pmo.block,
        name=None,
    ):
        self.name = name

        self.constraint_strings = {}
        self.sub_reports = {}

        # Find all children from within this block.
        for c in model.children():
            cName = c.local_name if hasattr(c, "local_name") else str(c)
            fullName = c.name
            try:
                obj = getattr(model, cName)
            except Exception:
                if ".DCC_constraint" in cName:
                    continue
                warnings.warn(f'Warning! Could not locate child object named "{c}"')
                continue

            if isinstance(
                obj,
                (
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
                    # pmo.sos1, #TODO: Should SOS's be given consideration here?
                    # pmo.sos2,
                    # pmo.sos_dict,
                    # pmo.sos_list,
                    # pmo.sos_tuple
                ),
            ):
                continue
            elif isinstance(obj, pmo.variable):
                if obj.is_fixed():
                    self.AddFixedVariable(obj.local_name, pmo.value(obj))
            elif isinstance(obj, (pmo.variable_list, pmo.variable_tuple)):
                for index in range(len(obj)):
                    if obj[index].is_fixed():
                        self.AddFixedVariable(
                            f"{obj.local_name}[{index}]", pmo.value(obj[index])
                        )
            elif isinstance(obj, pmo.variable_dict):
                for index in obj:
                    if obj[index].is_fixed():
                        self.AddFixedVariable(
                            f"{obj.local_name}[{index}]", pmo.value(obj[index])
                        )
            elif isinstance(obj, (pmo.constraint_list, pmo.constraint_tuple)):
                for index in range(len(obj)):
                    if obj[index].active:
                        self.AddConstraint(
                            name=f"{c.local_name}[{index}]", constr=obj[index]
                        )
            elif isinstance(obj, pmo.constraint_dict):
                for index in obj:
                    if obj[index].active:
                        self.AddConstraint(
                            name=f"{c.local_name}[{index}]", constr=obj[index]
                        )
            elif isinstance(obj, pmo.constraint):
                if obj.active:
                    self.AddConstraint(name=c.local_name, constr=obj)

            elif isinstance(obj, (pmo.block_list, pmo.block_tuple)):
                for index in range(len(obj)):
                    subName = f"{cName}[{index}]"
                    subReport = ConstraintReport(
                        obj[index],
                        name=subName,
                    )
                    self.sub_reports[subName] = subReport
            elif isinstance(obj, pmo.block_dict):
                for index in obj:
                    subName = f"{cName}[{index}]"
                    subReport = ConstraintReport(
                        obj[index],
                        name=subName,
                    )
                    self.sub_reports[subName] = subReport
            elif isinstance(obj, pmo.block):
                subName = cName
                subReport = ConstraintReport(
                    obj,
                    name=subName,
                )
                self.sub_reports[subName] = subReport
            else:
                pass

    def AddFixedVariable(self, name: str, value: float):
        self.constraint_strings[name] = f"{name} == {value} (Fixed Variable)"

    def AddConstraint(self, name: str, constr: pmo.constraint):
        self.constraint_strings[name] = str(constr.expr)

    def to_string(self, recursionDepth=1):
        """
        A function to convert this report to a string.
        """

        leftPad = "| " * (recursionDepth - 1)
        lines = [
            leftPad + (self.name if self.name is not None else "ROOT"),
        ]
        leftPad += "| "
        for constr_name, constr_str in self.constraint_strings.items():
            lines.append(f"{leftPad}{constr_name}: {constr_str}")

        result = "\n".join(lines)

        for sub_report_name, sub_report in self.sub_reports.items():
            sub_report_str = sub_report.to_string(recursionDepth + 1)
            result += f"\n{leftPad}{sub_report_name}:\n{sub_report_str}"

        return result

    def __str__(self):
        """
        A function to convert this report to a string.

        Usage
        -----
        result = str(reportObject)
        """
        return self.to_string()

    def WriteFile(self, fileName: str):
        """
        A function to write the output to a file.
        """
        with open(fileName, "w", encoding="utf-8") as f:
            f.write(self.to_string())

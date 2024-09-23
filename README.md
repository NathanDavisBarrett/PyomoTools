# PyomoTools
**PyomoTools** is a collection of tools to aid in formulating and working with [Pyomo](http://www.pyomo.org/) models.

Key functions include:

* **LoadIndexedSet**: Create a dict of subsets and properly attach it to a given pyomo model.
* **Load2DIndexedSet**: Similar to LoadIndexedSet but with two levels of indexing.
* **GenerateExpressionStrings**: A function to create a matching pair of strings representing a given pyomo expression: One symbolic, One with all values substituted in
* **InfeasibilityReport**: A class to analyze any infeasibilities found within a model in an easily readable way.
* **Formulations.DoubleSidedBigM**: A function to automatically generate the constraints and variables needed to model the relation $A = B \times X + C$ in MILP form.
* **Formulations.MinOperator**: A function to automatically generate the constraints and variables needed to model the relation $A = min(B,C)$ in MILP form.
* **Formulations.MaxOperator**: A function to automatically generate the constraints and variables needed to model the relation $A = max(B,C)$ in MILP form.
* **Formulations.PWL**: A function to automatically generate a Piecewise-Linear approximation of a general (non-linear) function
* **IO.ModelToExcel**: A function to write the solution of a model to an easily readable excel file.
* **IO.LoadModelSolutionFromExcel**: A function to load a solution from an excel file into a pyomo model.
* **MergeableModel**: A class that extends the base pyo.ConcreteModel class that now allows for sub-models to be added.
* **FindLeastInfeasibleSolution**: A tool for finding the least infeasible solution of a (presumably infeasible) model. 

# Installation
1. Download or clone this repository
2. In your Python terminal, navigate to the repository you downloaded.
3. Enter the command ```pip install .```
4. PyomoTools and all dependencies should be automatically installed.
5. To make sure everything was correctly installed, Enter the command ```pytest PyomoTools/```
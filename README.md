# PyomoTools
**PyomoTools** is a collection of tools to aid in formulating and working with [Pyomo](http://www.pyomo.org/) models.

Key functions include:

* **LoadIndexedSet**: Create a dict of subsets and properly attach it to a given pyomo model. (available to environ models only)
* **Load2DIndexedSet**: Similar to LoadIndexedSet but with two levels of indexing. (available to environ models only)
* **GenerateExpressionStrings**: A function to create a matching pair of strings representing a given pyomo expression: One symbolic, One with all values substituted in
* **InfeasibilityReport**: A class to analyze any infeasibilities found within a model in an easily readable way.
* **Formulations.DoubleSidedBigM**: A function to automatically generate the constraints and variables needed to model the relation $A = B \times X + C$ in MILP form.
* **Formulations.MinOperator**: A function to automatically generate the constraints and variables needed to model the relation $A = min(B,C)$ in MILP form.
* **Formulations.MaxOperator**: A function to automatically generate the constraints and variables needed to model the relation $A = max(B,C)$ in MILP form.
* **Formulations.PWL**: A function to automatically generate a Piecewise-Linear approximation of a general (non-linear) function (available to environ models only) (pyomo kernel already has this functionality)
* **IO.ModelToExcel**: A function to write the solution of a model to an easily readable excel file. (available to environ models only)
* **IO.LoadModelSolutionFromExcel**: A function to load a solution from an excel file into a pyomo model. (available to environ models only)
* **IO.ModelToYaml**: A function to write the solution of a model to a yaml file. (available to kernel models only)
* **IO.LoadSolutionFromYaml**: A function to load a solution from a yaml file into a pyomo model. (available to kernel models only)
* **FindLeastInfeasibleSolution**: A tool for finding the least infeasible solution of a (presumably infeasible) model. 
* **VectorRepresentation**: A tool to convert a (Mixed-Integer) Linear model into it's vector/matrix representation.
* **Polytope**: A class to facilitate the plotting and vertex calculation of a sub-polytope of a model.

Each function is available (or will soon be available) for both pyomo.environ modeling as well as pyomo.kernel modeling. To access each one, please import them from PyomoTools.environ or PyomoTools.kernel

# Installation
1. Download or clone this repository
2. In your Python terminal, navigate to the repository you downloaded.
3. By default, the example/testing code used into this package uses the [SCIP solver](https://github.com/scipopt/scip). Please ensure you have this solver installed ([Instalation Link](https://www.scipopt.org/index.php#download)). If you'd like to instead use a different solver(s), please edit the "Solvers.py" file to point to the solvers you'd like to use before continuing.
4. Enter the command ```pip install .```
5. PyomoTools and all dependencies should be automatically installed.
6. To make sure everything was correctly installed, Enter the command ```pytest PyomoTools/```
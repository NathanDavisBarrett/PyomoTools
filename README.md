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
### From PyPi
> 1.1. Enter the command ```pip install PyomoTools``` in your Python terminal.
### From GitHub
> 1.1. Download or clone this repository.\
> 1.2. In your Python terminal, navigate to the repository you downloaded.\
> 1.3. Enter the command ```pip install .```

2. By default, the example/testing code used into this package uses the [SCIP solver](https://github.com/scipopt/scip), [Gurobi Solver](https://www.gurobi.com/), and [ipopt Solver](https://coin-or.github.io/Ipopt/). Please either \
A) ensure that these solvers are installed and accessible to Pyomo or\
B) call ```PyomoTools.base.Solvers.DefaultSolver.EditDefaultSolver(problemType,solverName,**defaultKwargs)``` where\
```problemType``` is MILP, NLP, MIQCP, etc,\ ```solverName``` is the name of the solver you'd like to use (e.g. "gurobi", "ipopt"), and\ ```defaultKwargs``` are any necessary key-word arguments for pyomo to utilize (e.g. executable=path/to/executable)

3. To make sure everything was correctly installed, Enter the command ```pytest PyomoTools/```
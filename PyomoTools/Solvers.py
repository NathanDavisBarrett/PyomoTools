import pyomo.environ as pyo


def DefaultSolver(problemType="MILP"):
    """
    A function to return the key work arguments to use to get access to the default solver you'd like to use throughout this package depending of problem type.

    These are the default solvers and their configuration for my own machine you'll almost certainly need to change these defaults for your machine.

    Parameters:
    problemType: str (optional, Default)
        The problem type you would like the default solver for. Options are "LP", "MILP","QP","MIQP","MIQCP","NLP","MINLP"
    """
    if problemType in ["MILP","QP","MIQP","LP"]:
        return pyo.SolverFactory("scip")
    elif problemType == "NLP":
        return pyo.SolverFactory("scip")
    elif problemType in ["MINLP","MIQCP"]:
        return pyo.SolverFactory("scip")
    else:
        raise Exception(f"Problem Type \"{problemType}\" is not recognized.")
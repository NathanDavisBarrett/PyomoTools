import pyomo.environ as pyo


def DefaultSolver(problemType="MILP"):
    """
    A function to return the key work arguments to use to get access to the default solver you'd like to use throughout this package depending of problem type.

    These are the default solvers and their configuration for my own machine you'll almost certainly need to change these defaults for your machine.

    Parameters:
    problemType: str (optional, Default)
        The problem type you would like the default solver for. Options are "LP", "MILP","QP","MIQP","MIQCP","NLP","MINLP"
    """
    if problemType in ["MILP","QP","Convex MIQP","MIQCP"]:
        return pyo.SolverFactory("gurobi")
    elif problemType == "NLP":
        return pyo.SolverFactory("ipopt",executable=r"C:\Users\Public\ipopt.exe")
    elif problemType == "MINLP":
        return pyo.SolverFactory("baron",executable=r"C:\GAMS\41\baron.exe")
    else:
        raise Exception(f"Problem Type \"{problemType}\" is not recognized.")
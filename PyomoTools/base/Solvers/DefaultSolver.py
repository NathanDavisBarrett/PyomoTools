import pyomo.environ as pyo
import json
import os
from pathlib import Path


def get_config_path():
    """Get the path to the user configuration directory."""
    if os.name == "nt":  # Windows
        config_dir = Path(
            os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")
        )
    else:  # Unix-like (Linux, macOS)
        config_dir = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))

    pyomo_config_dir = config_dir / "PyomoTools"
    pyomo_config_dir.mkdir(exist_ok=True)
    return pyomo_config_dir / "solver_config.json"


def load_solver_config():
    """Load solver configuration from file, creating defaults if needed."""
    config_path = get_config_path()

    default_config = {
        "MILP": ("gurobi", {}),
        "QP": ("gurobi", {}),
        "MIQP": ("gurobi", {}),
        "LP": ("gurobi", {}),
        "NLP": ("ipopt", {}),
        "MIQCP": ("gurobi", {}),
        "MINLP": ("scip", {}),
    }

    if not config_path.exists():
        # Create default config file
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        return default_config

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is corrupted, recreate with defaults
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        return default_config


def EditDefaultSolver(problemType, solver_name, **kwargs):
    """
    Edit the default solver for a given problem type.
    Configuration is saved to user config file.

    Parameters:
    problemType (str): The type of problem (e.g., "MILP", "NLP", etc.).
    solver_name (str): The name of the solver to use (e.g., "gurobi", "ipopt").
    **kwargs: Additional keyword arguments to pass to the solver (e.g. "executable": path/to/executable).
    """
    config = load_solver_config()
    config[problemType] = (solver_name, kwargs)

    config_path = get_config_path()
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def DefaultSolver(problemType="MILP"):
    """
    A function to return the solver for the specified problem type.
    Configuration is loaded from user config file.
    """
    config = load_solver_config()

    if problemType not in config:
        raise Exception(
            f'Problem Type "{problemType}" is not recognized. Recognized types are: {list(config.keys())}'
        )

    solver_name, kwargs = config[problemType]
    return pyo.SolverFactory(solver_name, **kwargs)

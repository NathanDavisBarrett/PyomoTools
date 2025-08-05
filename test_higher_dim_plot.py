#!/usr/bin/env python3
"""
Test script for the _PlotHigherDim function
"""

import pyomo.kernel as pmo
import numpy as np
from PyomoTools.kernel.MatrixRepresentation.MatrixRepresentation import MatrixRepresentation

def create_test_model():
    """Create a simple test model with 4+ variables for testing higher dimensional plotting"""
    
    model = pmo.block()
    
    # Create variables
    model.x1 = pmo.variable(lb=0, ub=10)
    model.x2 = pmo.variable(lb=0, ub=10)
    model.x3 = pmo.variable(lb=0, ub=10)
    model.x4 = pmo.variable(lb=0, ub=10)
    model.x5 = pmo.variable(lb=0, ub=10)
    
    # Add some constraints to create a bounded polytope
    model.c1 = pmo.constraint(model.x1 + model.x2 + model.x3 + model.x4 + model.x5 <= 20)
    model.c2 = pmo.constraint(2*model.x1 + model.x2 - model.x3 <= 15)
    model.c3 = pmo.constraint(-model.x1 + 3*model.x2 + model.x4 <= 25)
    model.c4 = pmo.constraint(model.x1 - model.x2 + 2*model.x3 - model.x5 <= 10)
    model.c5 = pmo.constraint(model.x3 + model.x4 + model.x5 >= 5)
    
    # Add objective (optional)
    model.obj = pmo.objective(model.x1 + 2*model.x2 + model.x3 - model.x4 + 0.5*model.x5, sense=pmo.minimize)
    
    return model

def test_higher_dim_plot():
    """Test the higher dimensional plotting functionality"""
    
    print("Creating test model with 5 variables...")
    model = create_test_model()
    
    print("Creating matrix representation...")
    matrix_rep = MatrixRepresentation(model)
    
    print(f"Model has {len(matrix_rep.VAR_VEC)} variables and {len(matrix_rep.CONSTR_VEC)} constraints")
    print(f"Found {len(matrix_rep.vertices)} vertices")
    
    if len(matrix_rep.vertices) > 0:
        print("Sample vertices:")
        for i, vertex in enumerate(matrix_rep.vertices[:5]):  # Show first 5 vertices
            print(f"  Vertex {i+1}: {vertex}")
        if len(matrix_rep.vertices) > 5:
            print(f"  ... and {len(matrix_rep.vertices) - 5} more vertices")
    
    print("\nLaunching interactive higher dimensional plot...")
    print("Instructions:")
    print("- Use the checkboxes to select which variables to display on X, Y, and Z axes")
    print("- Use the sliders to set values for the non-displayed variables")
    print("- The plot will show the 3D projection of your polytope")
    
    # Launch the interactive plot
    matrix_rep.Plot()

if __name__ == "__main__":
    test_higher_dim_plot()

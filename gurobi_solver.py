from gurobipy import Model, GRB, quicksum
import numpy as np


def optimize_q(r_ij, T):
    n, K = r_ij.shape # Number of nodes and configurations
    model = Model("fair_allocation") 

    # Variables
    q = model.addVars(K, vtype=GRB.INTEGER, lb=0, name="q") # q[j]: how many days to use configuration j
    y = model.addVars(n, vtype=GRB.CONTINUOUS, name="y") # y[i]: average service for node i
    g = model.addVar(vtype=GRB.CONTINUOUS, name="g") # g: maximum of y[i]
    f = model.addVar(vtype=GRB.CONTINUOUS, name="f") # f: minimum of y[i]

    # Objective
    model.setObjective(g - f, GRB.MINIMIZE) # Minimize the gap between max and min average service

    # Total config usage = T days
    model.addConstr(quicksum(q[j] for j in range(K)) == T, "total_days") # Total days constraint

    # Define y_i = (1/T) * sum_j r_ij * q_j
    for i in range(n):
        # y[i] = (1/T) * sum_j r_ij[i][j] * q[j]
        model.addConstr(
            y[i] == (1.0 / T) * quicksum(r_ij[i][j] * q[j] for j in range(K)),
            f"y_def_{i}"
        )
        model.addConstr(g >= y[i], f"g_bound_{i}") # g >= y[i]
        model.addConstr(f <= y[i], f"f_bound_{i}") # f <= y[i]

    model.setParam("OutputFlag", 0) 
    model.optimize()

    # Check if the model has an optimal solution
    if model.status == GRB.OPTIMAL:
        q_vals = [q[j].X for j in range(K)] # Optimal values of q[j]
        y_vals = [y[i].X for i in range(n)] # Optimal values of y[i]
        return q_vals, y_vals, g.X, f.X, g.X - f.X # Return optimal q, y, g, f, and gap
    else:
        return None
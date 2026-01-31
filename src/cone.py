import numpy as np
import gurobipy as gp
from gurobipy import GRB


def _check_cone_nullity(
    H: np.ndarray,  # assume pointed
    verbose: bool = False,
):
    model = gp.Model("check_cone_nullity")
    model.Params.OutputFlag = 1 if verbose else 0

    # Decision variables
    y = model.addMVar(shape=H.shape[0], lb=1, name=f"y0")
    eps = model.addVar(lb=0, name="eps")
    
    # Constraints: | y^T H | <= eps:
    model.addConstr(y.T @ H - eps <= 0)
    model.addConstr(y.T @ H + eps >= 0)

    # Objective: minimize eps
    model.setObjective(eps, GRB.MINIMIZE)

    model.optimize()

    status = model.Status
    if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return None, None, status

    y_val = y.X
    eps_val = float(eps.X)
    return y_val, eps_val, status


# end of cone.py
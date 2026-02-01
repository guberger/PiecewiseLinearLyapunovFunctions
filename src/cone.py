import numpy as np
import gurobipy as gp
from gurobipy import GRB


def _check_cone_nullity(
    H: np.ndarray,  # assume pointed
    verbose: bool = False,
):
    """
    Checks whether the cone ``H @ x >= 0`` contains only the origin.

    The test is performed by searching for a vector ``y >= 1`` that minimizes
    ``eps = || y.T @ H ||_∞``. The cone ``H @ x >= 0`` equals ``{0}`` if the
    optimal value ``eps`` is zero (up to numerical tolerance).

    Parameters
    ----------
    H : np.ndarray
        Matrix defining the cone via ``H @ x >= 0``. The cone is assumed to be pointed.
    verbose : bool, optional
        If True, print details from the LP solver.

    Returns
    -------
    eps : float
        Optimal value of the LP. The cone equals ``{0}`` if ``eps == 0`` up to rounding error.
    """

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
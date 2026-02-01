import numpy as np
import gurobipy as gp
from gurobipy import GRB


def _check_cone_nullity(
    H: np.ndarray,  # assume pointed
    verbose: bool = False,
):
    """
    Check whether a polyhedral cone is trivial (contains only the origin).

    This function tests whether the cone defined by

        H @ x >= 0

    contains only the zero vector. The test is based on the dual characterization:
    it searches for a vector ``y >= 1`` that minimizes

        eps = || y.T @ H ||_∞.

    If the optimal value ``eps`` is zero (up to numerical tolerance), then the cone
    equals ``{0}``. The optimization problem is solved using a linear program.

    Parameters
    ----------
    H : np.ndarray
        Two-dimensional array defining the cone via ``H @ x >= 0``.
        The cone is assumed to be pointed.
    verbose : bool, optional
        If True, enable solver output. Default is False.

    Returns
    -------
    y : np.ndarray or None
        Optimal value of the dual variable ``y`` if the solver terminates with
        status OPTIMAL or SUBOPTIMAL. Otherwise, None.
    eps : float or None
        Optimal objective value ``eps`` if the solver terminates with
        status OPTIMAL or SUBOPTIMAL. Otherwise, None.
    status : int
        Solver termination status code.

    Notes
    -----
    The cone is considered equal to ``{0}`` if ``eps`` is numerically zero.
    In practice, a small tolerance should be used when interpreting the result.

    This function formulates the constraints

        -eps <= y.T @ H <= eps,
        y >= 1,

    and minimizes ``eps`` subject to these constraints.
    """

    model = gp.Model("check_cone_nullity")
    model.Params.OutputFlag = 1 if verbose else 0

    # Decision variables
    y = model.addMVar(shape=H.shape[0], lb=1, name=f"y0")
    eps = model.addVar(lb=0, name="eps")
    
    # Constraints: | y.T @ H | <= eps:
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
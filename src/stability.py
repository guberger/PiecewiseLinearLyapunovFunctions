import numpy as np
import gurobipy as gp
from gurobipy import GRB
from src.system import Path, Edge, PiecewiseLinearSystem, _build_path_domain


def compute_lyapunov(
    sys: PiecewiseLinearSystem,
    node_list: list[Path],
    edge_list: list[Edge],
    verbose: bool = False,
):
    """
    Computes a piecewise-linear Lyapunov function for the piecewise-linear system ``sys`` using Linear Programming.

    Each piece is defined on a conic domain ``H @ x >= 0`` with value ``c.T @ x``. The i-th piece corresponds to
    ``(H_list[i], c_list[i])``. The computation is considered successful if ``eps > 0``.

    Parameters
    ----------
    sys : PiecewiseLinearSystem
        Piecewise linear system.
    node_list : list[Path]
        List of nodes of the path graph.
    edge_list : list[Edge]
        List of edges of the path graph.
    verbose : bool, optional
        If True, print details from the LP solver.

    Returns
    -------
    H_list : list[np.ndarray]
        List of constraint matrices. ``H_node_list[i]`` defines the conic domain of piece i via ``H @ x >= 0``.
    c_list : list[np.ndarray]
        List of coefficient vectors. ``c_list[i]`` defines the linear function of piece i via ``c.T @ x``.
    eps : float
        Feasibility / margin value. The computation is successful if ``eps > 0``.
    """
    
    if len(node_list) == 0:
        raise ValueError("At least one node is required.")
    
    H_node_list = []
    for node in node_list:
        H = _build_path_domain(sys, node)
        H_node_list.append(H)

    H_edge_list = []
    for (k0, k1, _) in edge_list:
        node0 = node_list[k0]
        node1 = node_list[k1]
        assert node0[1:] == node1[:-1]
        path = node0.copy()
        path.append(node1[-1])
        H = _build_path_domain(sys, path)
        H_edge_list.append(H)

    model = gp.Model("compute_pwl_lyapunov")
    model.Params.OutputFlag = 1 if verbose else 0

    # Decision variables
    y_list = [
        model.addMVar(shape=H.shape[0], ub=1e3, name=f"y_{k}")
        for (k, H) in enumerate(H_node_list)
    ]
    z_list = [
        model.addMVar(shape=H.shape[0], ub=1e3, name=f"z_{r}")
        for (r, H) in enumerate(H_edge_list)
    ]
    eps = model.addVar(ub=2e3, name="eps")

    # Lower bound on y and z
    for y in y_list:
        model.addConstr(y >= eps)
    for z in z_list:
        model.addConstr(z >= eps)

    # Constraint edge (k0, k1, i): c_k1^T A x <= c_k0^T x for all H_edge x >= 0
    # where c_k^T = y_k^T H_k
    for ((k0, k1, label), z, H_edge) in zip(edge_list, z_list, H_edge_list):
        H0 = H_node_list[k0]
        H1 = H_node_list[k1]
        y0 = y_list[k0]
        y1 = y_list[k1]
        A = sys.A_list[label]
        model.addConstr(y0.T @ H0 - y1.T @ H1 @ A == z.T @ H_edge)

    # Objective: maximize eps
    model.setObjective(eps, GRB.MAXIMIZE)

    model.optimize()

    status = model.Status
    if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return None, None, None, status

    c_list = [y.X.T @ H_node for (H_node, y) in zip(H_node_list, y_list)]
    eps_val = float(eps.X)

    return H_node_list, c_list, eps_val, status


# end of stability.py
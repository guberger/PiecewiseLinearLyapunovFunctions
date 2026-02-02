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
    Compute a piecewise-linear Lyapunov function via linear programming.

    This function attempts to construct a piecewise-linear Lyapunov function for
    a given piecewise-linear system using the path graph representation. Each
    Lyapunov piece is defined on a conic domain

        H @ x >= 0,

    with value function

        V(x) = c.T @ x.

    The conic domains correspond to the nodes of the path graph, and the Lyapunov
    decrease conditions are enforced along the edges of the graph. The Lyapunov
    function is certified if the optimal margin ``eps`` returned by the linear
    program is strictly positive.

    Parameters
    ----------
    sys : PiecewiseLinearSystem
        Piecewise-linear system defining the dynamics.
    node_list : list[Path]
        List of nodes of the path graph. Each node defines a conic domain via
        its associated path.
    edge_list : list[Edge]
        List of directed edges of the path graph encoding admissible transitions
        between nodes.
    verbose : bool, optional
        If True, enable solver output from the underlying linear program.
        Default is False.

    Returns
    -------
    H_list : list[np.ndarray]
        List of constraint matrices defining the conic domains of the Lyapunov
        pieces. The i-th matrix ``H_list[i]`` defines the domain via
        ``H_list[i] @ x >= 0``.
    c_list : list[np.ndarray]
        List of coefficient vectors defining the Lyapunov function pieces.
        The i-th Lyapunov function is ``V_i(x) = c_list[i].T @ x``.
    eps : float
        Optimal margin value returned by the linear program. The construction
        is successful if ``eps > 0``.
    status : int
        Solver termination status code.

    Notes
    -----
    The Lyapunov coefficients are constructed indirectly via dual variables
    ``y_k`` associated with each node domain, with

        c_k.T = y_k.T @ H_k.
    
    In this ``c_k.T @ x`` is positive on ``H_k @ x > 0``.
    For each edge ``(k0, k1, i)``, the linear program enforces the decrease
    condition

        c_{k1}.T @ A_i @ x <= c_{k0}.T @ x

    for all ``x`` in the conic domain associated with the concatenated path.
    No normalization of the Lyapunov function is imposed beyond the bounds
    encoded in the optimization problem.
    """

    if len(node_list) == 0:
        raise ValueError("At least one node is required.")
    
    H_node_list = []
    for node in node_list:
        H = _build_path_domain(sys, node)
        H_node_list.append(H)

    H_edge_list = []
    for edge in edge_list:
        node0 = node_list[edge.src]
        node1 = node_list[edge.tar]
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

    # Lower bounds on y and z
    for y in y_list:
        model.addConstr(y >= eps)
    for z in z_list:
        model.addConstr(z >= eps)

    # Constraints:
    # For each edge ``(k0, k1, i)``, the linear program enforces the decrease
    # condition ``c_{k1}.T @ A_i @ x <= c_{k0}.T @ x`` on the conic domain
    # ``H_edge @ x >= 0``.
    for (edge, z, H_edge) in zip(edge_list, z_list, H_edge_list):
        H0 = H_node_list[edge.src]
        H1 = H_node_list[edge.tar]
        y0 = y_list[edge.src]
        y1 = y_list[edge.tar]
        A = sys.A_list[edge.lab]
        model.addConstr(y0.T @ H0 - y1.T @ H1 @ A == z.T @ H_edge)

    # Objective
    model.setObjective(eps, GRB.MAXIMIZE)

    model.optimize()

    status = model.Status
    if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return None, None, None, status

    c_list = [y.X.T @ H_node for (H_node, y) in zip(H_node_list, y_list)]
    eps_val = float(eps.X)

    return H_node_list, c_list, eps_val, status


# end of stability.py
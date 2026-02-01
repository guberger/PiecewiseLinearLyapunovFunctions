import itertools
import numpy as np
from src.cone import _check_cone_nullity


type Path = list[int]
type Edge = tuple[int,int,int]


class PiecewiseLinearSystem:
    """
    Piecewise-linear system of dimension ``n``.

    Each piece is defined on a conic domain ``H @ x >= 0`` with linear dynamics
    ``x ↦ A @ x``. The i-th piece of the system corresponds to ``(H_list[i], A_list[i])``.

    Attributes
    ----------
    H_list : list[np.ndarray]
        List of matrices defining the conic domains via ``H @ x >= 0``.
    A_list : list[np.ndarray]
        List of system matrices defining the linear dynamics ``A @ x`` on each domain.
    n : int
        State-space dimension.
    """

    def __init__(
        self,
        H_list: list[np.ndarray],
        A_list: list[np.ndarray],
    ):
        if len(H_list) == 0 or len(A_list) == 0:
            raise ValueError("At least one subsystem is required.")
        if not (len(A_list) == len(H_list)):
            raise ValueError("H_list and A_list must have the same length.")
        self.H_list = [np.asarray(H, dtype=float) for H in H_list]
        self.A_list = [np.asarray(A, dtype=float) for A in A_list]
        if self.H_list[0].ndim != 2:
            raise ValueError(f"H_0 must be 2D.")
        self.n = self.H_list[0].shape[1]
        # Dimension check:
        for (i, H) in enumerate(self.H_list):
            if i == 0:
                continue
            if H.ndim != 2:
                raise ValueError(f"H_{i} must be 2D.")
            if H.shape[1] != self.n:
                raise ValueError(f"H_{i} has incompatible shape.")
        for (i, A) in enumerate(self.A_list):
            if A.shape != (self.n, self.n):
                raise ValueError(f"As[{i}] has incompatible shape.")


def _build_path_domain(
    sys: PiecewiseLinearSystem,
    path: Path,
):
    """
    Constructs the polyhedral cone of initial states consistent with a given path.

    Given a sequence ``path`` of ``m`` pieces of ``sys``, this function builds the
    polyhedral cone ``H @ x >= 0`` consisting of all initial states ``x`` for which
    there exists a trajectory ``x_traj`` such that
    ``x_traj[i]`` lies in the domain of ``path[i]`` for all ``0 <= i < m``.

    Parameters
    ----------
    sys : PiecewiseLinearSystem
        Piecewise-linear system.
    path : Path
        Non-empty sequence of pieces of ``sys``.

    Returns
    -------
    H : np.ndarray
        Matrix defining the polyhedral cone of admissible initial states via ``H @ x >= 0``.
    """

    if len(path) == 0:
        raise ValueError("At least one step is required.")
    H = sys.H_list[path[0]]
    B = sys.A_list[path[0]]
    for (k, i) in enumerate(path):
        if k == 0:
            continue
        H = np.vstack((H, sys.H_list[i] @ B))
        if k == len(path) - 1:
            continue
        B = sys.A_list[i] @ B
    return H


def _check_path_feasibility(
    sys: PiecewiseLinearSystem,
    path: Path,
    tol: float = 1e-8,
    verbose: bool = False,
):
    """
    Checks feasibility of a path by testing nullity of its domain.

    The domain associated with ``path`` is constructed as a polyhedral cone.
    If the cone is null (up to tolerance ``tol``), the path is declared infeasible.

    Parameters
    ----------
    sys : PiecewiseLinearSystem
        Piecewise-linear system.
    path : Path
        Sequence of pieces of ``sys``.
    tol : float
        Tolerance for infeasibility. The path is infeasible if ``eps < tol``.
    verbose : bool, optional
        If True, print solver details from the underlying feasibility check.

    Returns
    -------
    feasible : bool
        True if the path domain is non-null up to tolerance ``tol``,
        False otherwise.
    eps : float
        Value used to test nullity of the path domain.
    """

    H = _build_path_domain(sys, path)    
    y, eps, status = _check_cone_nullity(H, verbose=verbose)
    if y is None:
        raise ValueError(f"Status: {status}")
    return eps > tol


def _compute_feasible_path_list(
    sys: PiecewiseLinearSystem,
    length: int,
    tol: float = 1e-8,
    verbose: bool = False,
):
    """
    Computes all feasible paths of a given length for a piecewise-linear system.

    A path is deemed infeasible if the nullity check of its domain yields
    ``eps < tol``, where ``eps`` is the output of the cone nullity test.
    This implementation uses a recursive procedure.

    Parameters
    ----------
    sys : PiecewiseLinearSystem
        Piecewise-linear system.
    length : int
        Length of the paths.
    tol : float
        Tolerance for infeasibility. A path is infeasible if ``eps < tol``.
    verbose : bool, optional
        If True, print solver details from the underlying feasibility checks.

    Returns
    -------
    paths : list[Path]
        List of all feasible paths of length ``length``.
    """

    old_feasible_path_list = [[]]
    for _ in range(length):
        feasible_path_list = []
        for old_node in old_feasible_path_list:
            for i in range(len(sys.H_list)):
                path = old_node.copy()
                path.append(i)
                if _check_path_feasibility(sys, path, verbose):
                    feasible_path_list.append(path)
        old_feasible_path_list = feasible_path_list
    return feasible_path_list


# def _compute_feasible_path_list_old(
#     sys: PiecewiseLinearSystem,
#     length: int,
#     tol: float = 1e-8,
#     verbose: bool = False,
# ):
#     if not isinstance(length, int) or length <= 0:
#         raise ValueError("Length must be a positive integer")
#     n_sub = len(sys.H_list)
#     feasible_path_list = []
#     for _path_ in itertools.product(range(n_sub), repeat=length):
#         path = list(_path_)
#         y, eps, status = _check_path_feasibility(sys, path, verbose)
#         if y is None:
#             raise ValueError(f"Status: {status}")
#         if eps > tol:
#             feasible_path_list.append(path)
#     return feasible_path_list


def _find_or_add(node_list: list[Path], target_node: Path):
    """
    Finds the index of a node in a list, adding it if absent.

    Parameters
    ----------
    node_list : list[Path]
        List of nodes.
    target_node : Path
        Node to search for.

    Returns
    -------
    index : int
        Index of ``target_node`` in ``node_list`` after insertion if needed.
    """
    
    for (k, node) in enumerate(node_list):
        if node == target_node:
            return k, True
    node_list.append(target_node)
    return len(node_list) - 1, False


def compute_path_graph(
    sys: PiecewiseLinearSystem,
    length: int,
    tol: float = 1e-8,
    verbose: bool = False,
):
    """
    Builds the path graph associated with a piecewise-linear system.

    Nodes of the graph correspond to feasible paths of length ``length``.
    There is a directed edge from ``node0`` to ``node1`` if
    ``node0[1:] == node1[:-1]`` and the concatenated path
    ``node0 + [node1[-1]]`` is feasible.

    Parameters
    ----------
    sys : PiecewiseLinearSystem
        Piecewise-linear system.
    length : int
        Length of the paths used as graph nodes.
    tol : float
        Tolerance for infeasibility. A path is infeasible if ``eps < tol``.
    verbose : bool, optional
        If True, print solver details from the underlying feasibility checks.

    Returns
    -------
    node_list : list[Path]
        List of nodes of the path graph (feasible paths of length ``length``).
    edge_list : list[Edge]
        List of directed edges of the path graph.
    """

    if not isinstance(length, int) or length <= 0:
        raise ValueError("Length must be a positive integer")
    
    path_list = _compute_feasible_path_list(sys, length + 1, tol, verbose)
    
    node_list = []
    edge_list = []
    
    for path in path_list:
        node0 = path[:-1]
        k0, _ = _find_or_add(node_list, node0)
        node1 = path[1:]
        k1, _ = _find_or_add(node_list, node1)
        label = path[0]
        edge_list.append((k0, k1, label))
    
    return node_list, edge_list


# end of system.py
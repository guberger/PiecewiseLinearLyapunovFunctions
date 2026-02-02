import numpy as np
from src.cone import _check_cone_nullity


type Path = list[int]
type Edge = tuple[int,int,int]


class PiecewiseLinearSystem:
    """
    Piecewise-linear dynamical system defined on conic regions.

    This class represents a discrete-time piecewise-linear system of the form

        x ↦ A_i @ x   if   H_i @ x >= 0,

    where each pair ``(H_i, A_i)`` defines a linear subsystem active on a conic
    domain. The full system consists of a finite collection of such subsystems,
    each defined on a (possibly overlapping) polyhedral cone.

    Parameters
    ----------
    H_list : list[np.ndarray]
        List of two-dimensional arrays defining the conic domains via
        ``H_i @ x >= 0``. All matrices must have the same number of columns,
        equal to the state dimension.
    A_list : list[np.ndarray]
        List of square system matrices defining the linear dynamics
        ``x ↦ A_i @ x`` on each corresponding domain. Each matrix must have
        shape ``(n, n)``.

    Attributes
    ----------
    H_list : list[np.ndarray]
        List of matrices defining the conic domains.
    A_list : list[np.ndarray]
        List of system matrices defining the linear dynamics on each domain.
    n : int
        State-space dimension.


    Notes
    -----
    The class does not enforce disjointness or completeness of the conic domains;
    it only stores the subsystem definitions and checks dimensional consistency.
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
    Construct the cone of initial states consistent with a given switching path.

    Given a finite path of subsystem indices, this function computes a matrix ``H``
    such that the polyhedral cone

        H @ x >= 0

    characterizes all initial states ``x`` for which there exists a trajectory
    ``x_traj`` satisfying the domain constraints induced by the path. Specifically,
    for a path ``(i_0, i_1, ..., i_{m-1})``, the trajectory must satisfy

        x_traj[0] = x,
        x_traj[k + 1] = A_{i_k} @ x_traj[k],
        H_{i_k} @ x_traj[k] >= 0   for all 0 <= k < m.

    The resulting cone is expressed entirely in terms of the initial state ``x``
    by propagating the domain constraints backward through the dynamics.

    Parameters
    ----------
    sys : PiecewiseLinearSystem
        Piecewise-linear system defining the conic domains and linear dynamics.
    path : Path
        Non-empty sequence of indices specifying a path through the pieces of
        ``sys``.

    Returns
    -------
    H : np.ndarray
        Two-dimensional array defining the polyhedral cone of admissible initial
        states via ``H @ x >= 0``.
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
    Check feasibility of a switching path via nullity of its domain.

    This function determines whether a given path through the pieces of a
    piecewise-linear system is feasible. Feasibility is tested by constructing
    the polyhedral cone of initial states consistent with the path and checking
    whether this cone is non-null.

    Let ``H @ x >= 0`` be the cone associated with ``path``. The path is declared
    infeasible if this cone reduces to ``{0}`` up to numerical tolerance. The
    nullity test is performed by solving an auxiliary optimization problem whose
    optimal value is returned as ``eps``.

    Parameters
    ----------
    sys : PiecewiseLinearSystem
        Piecewise-linear system defining the conic domains and linear dynamics.
    path : Path
        Sequence of subsystem indices specifying a path through ``sys``.
    tol : float, optional
        Numerical tolerance for infeasibility. The path is considered infeasible
        if ``eps <= tol``. Default is ``1e-8``.
    verbose : bool, optional
        If True, enable solver output from the underlying cone nullity check.
        Default is False.

    Returns
    -------
    feasible : bool
        True if the path domain is non-null up to tolerance ``tol``,
        False otherwise.

    Raises
    ------
    RuntimeError
        If the underlying cone nullity check fails to terminate with an acceptable
        solver status.
    """

    H = _build_path_domain(sys, path)    
    y, eps, status = _check_cone_nullity(H, verbose=verbose)
    if y is None:
        raise RuntimeError(f"Status: {status}")
    return eps > tol


def _compute_feasible_path_list(
    sys: PiecewiseLinearSystem,
    length: int,
    tol: float = 1e-8,
    verbose: bool = False,
):
    """
    Compute all feasible switching paths of a given length.

    This function enumerates all paths of a prescribed length through the pieces
    of a piecewise-linear system and retains only those that are feasible. A path
    is considered feasible if the polyhedral cone of initial states consistent
    with the path is non-null up to a specified numerical tolerance.

    Feasibility of each candidate path is determined by constructing its domain
    and applying a cone nullity test. Paths for which the optimal value ``eps``
    returned by this test satisfies ``eps <= tol`` are discarded.

    The enumeration is performed incrementally: starting from the empty path,
    feasible paths are extended one step at a time and pruned as soon as they
    become infeasible.

    Parameters
    ----------
    sys : PiecewiseLinearSystem
        Piecewise-linear system defining the conic domains and linear dynamics.
    length : int
        Desired length of the switching paths.
    tol : float, optional
        Numerical tolerance for infeasibility. A path is discarded if
        ``eps <= tol``. Default is ``1e-8``.
    verbose : bool, optional
        If True, enable solver output from the underlying feasibility checks.
        Default is False.

    Returns
    -------
    paths : list[Path]
        List of all feasible paths of length ``length``. Each path is represented
        as a list of subsystem indices.

    Notes
    -----
    The number of feasible paths may grow exponentially with ``length`` in the
    worst case. No memoization or symmetry reduction is performed.
    """

    old_feasible_path_list = [[]]
    for _ in range(length):
        feasible_path_list = []
        for old_node in old_feasible_path_list:
            for i in range(len(sys.H_list)):
                path = old_node.copy()
                path.append(i)
                if _check_path_feasibility(sys, path, tol, verbose):
                    feasible_path_list.append(path)
        old_feasible_path_list = feasible_path_list
    return feasible_path_list


# import itertools

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


def _get_index_with_insert(my_list: list, target):
    """
    Find the index of an object in a list, inserting it if absent.

    This utility function searches for ``target`` in ``my_list`` using
    equality comparison. If the target is already present, its index is returned.
    Otherwise, the target is appended to the list and the index of the newly added
    entry is returned.

    Parameters
    ----------
    my_list : list[Path]
        List of existing objects. The list is modified in place if
        ``target`` is not found.
    target : Path
        Object to search for.

    Returns
    -------
    index : int
        Index of ``target`` in ``my_list`` after insertion if needed.
    """

    index = next((l for (l, item) in enumerate(my_list) if item == target), -1)
    
    if index < 0:
        index = len(my_list)
        my_list.append(target)

    return index


def compute_path_graph(
    sys: PiecewiseLinearSystem,
    length: int,
    tol: float = 1e-8,
    verbose: bool = False,
):
    """
    Construct the path graph of a piecewise-linear system.

    This function builds a directed graph whose nodes correspond to feasible
    paths of a fixed length through the pieces of a piecewise-linear system.
    Edges encode feasible one-step extensions of these paths.

    Let ``length = L``. Nodes of the graph are all feasible paths of length ``L``.
    There is a directed edge from node ``p`` to node ``q`` if

        p[1:] == q[:-1]

    and the concatenated path ``p + [q[-1]]`` is feasible. Each edge is labeled
    by the first element of the concatenated path.

    The graph is constructed by enumerating all feasible paths of length
    ``L + 1`` and extracting their overlapping subpaths.

    Parameters
    ----------
    sys : PiecewiseLinearSystem
        Piecewise-linear system defining the conic domains and linear dynamics.
    length : int
        Length of the paths used as graph nodes. Must be a positive integer.
    tol : float, optional
        Numerical tolerance for infeasibility. A path is discarded if
        ``eps <= tol``. Default is ``1e-8``.
    verbose : bool, optional
        If True, enable solver output from the underlying feasibility checks.
        Default is False.

    Returns
    -------
    node_list : list[Path]
        List of nodes of the path graph. Each node is a feasible path of
        length ``length``.
    edge_list : list[Edge]
        List of directed edges of the path graph. Each edge is a tuple
        ``(k0, k1, label)``, where ``k0`` and ``k1`` are indices into
        ``node_list`` and ``label`` identifies the active piece for the
        transition.
    """

    if not isinstance(length, int) or length <= 0:
        raise ValueError("Length must be a positive integer")
    
    path_list = _compute_feasible_path_list(sys, length + 1, tol, verbose)
    
    node_list = []
    edge_list = []
    
    for path in path_list:
        node0 = path[:-1]
        k0 = _get_index_with_insert(node_list, node0)
        node1 = path[1:]
        k1 = _get_index_with_insert(node_list, node1)
        label = path[0]
        edge_list.append((k0, k1, label))
    
    return node_list, edge_list


# end of system.py
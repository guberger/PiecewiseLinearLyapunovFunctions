import itertools
import numpy as np
from src.cone import _check_cone_nullity


type Path = list[int]
type Edge = tuple[int,int,int]

class PiecewiseLinearSystem:
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
    path: list[int],
    verbose: bool = False,
):
    H = _build_path_domain(sys, path)    
    return _check_cone_nullity(H, verbose=verbose)


def _compute_feasible_path_list(
    sys: PiecewiseLinearSystem,
    length: int,
    tol: float = 1e-8,
    verbose: bool = False,
):
    if not isinstance(length, int) or length <= 0:
        raise ValueError("Length must be a positive integer")
    n_sub = len(sys.H_list)
    feasible_path_list = []
    for _path_ in itertools.product(range(n_sub), repeat=length):
        path = list(_path_)
        y, eps, status = _check_path_feasibility(sys, path, verbose)
        if y is None:
            raise ValueError(f"Status: {status}")
        if eps > tol:
            feasible_path_list.append(path)
    return feasible_path_list


def _find_or_add(node_list: list[Path], target_node: Path):
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
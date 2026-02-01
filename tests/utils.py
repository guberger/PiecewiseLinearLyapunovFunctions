import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection


# ---------- Geometry helpers ----------

def _chebyshev_center_and_radius(M: np.ndarray, b: np.ndarray):
    """
    Maximize r s.t. M x + ||M_i|| r <= -b  (with halfspaces M x + b <= 0)
    Here we pass constraints as: M x <= -b.
    """
    norm_vector = np.linalg.norm(M, axis=1, keepdims=True)

    # Decision vars: [x, r]
    c_obj = np.zeros(M.shape[1] + 1)
    c_obj[-1] = -1.0  # maximize r -> minimize -r

    A_ub = np.hstack([M, norm_vector])
    b_ub = -b

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=(None, None))
    if not res.success:
        return None, 0.0

    center = res.x[:-1]
    radius = res.x[-1]
    return center, radius


def _halfspaces_for_piece(H: np.ndarray, c: np.ndarray, level: float):
    """
    Build halfspaces for intersection:
      -H x <= 0     (i.e. H x >= 0)
       c^T x <= -level
    in the form A x + b <= 0 => (A, b)
    """
    c = np.asarray(c).reshape(-1, 1)
    M = np.vstack([-H, c.T])
    b = np.vstack([np.zeros((H.shape[0], 1)), -level * np.ones((1, 1))])
    return M, b


def _vertices_from_halfspaces(M: np.ndarray, b: np.ndarray, eps_center: float = 1e-8, eps_vert: float = 1e-6):
    center, radius = _chebyshev_center_and_radius(M, b)
    if center is None or radius < eps_center:
        return []

    # HalfspaceIntersection expects [A | b] with A x + b <= 0
    halfspaces = np.hstack([M, b])
    hs = HalfspaceIntersection(halfspaces, center)
    vertices = [v for v in hs.intersections if np.linalg.norm(v) > eps_vert]
    return vertices


# ---------- Plot helpers ----------

def plot_piece_levelset(ax, H: np.ndarray, c: np.ndarray, level: float = 1e3):
    M, b = _halfspaces_for_piece(H, c, level=level)
    vertices = _vertices_from_halfspaces(M, b)
    if not vertices:
        return None

    # Close the polygon-ish chain with the origin (assumes 2D plotting intent)
    vertices_aug = [np.zeros(2)] + vertices + [np.zeros(2)]
    x, y = zip(*vertices_aug)
    ax.plot(x, y, "-o", markersize=2)
    return vertices


def plot_all_piece_levelsets(ax, H_list, c_list, level: float = 1e3):
    all_vertices = []
    for H, c in zip(H_list, c_list):
        vertices = plot_piece_levelset(ax, H, c, level=level)
        if vertices is not None:
            all_vertices = all_vertices + vertices
    return all_vertices


# ---------- System / trajectory helpers ----------

def find_active_piece_index(H_list, x: np.ndarray, tol: float = 1e-8):
    for i, H in enumerate(H_list):
        if np.all(H @ x >= -tol):
            return i
    return None


def simulate_trajectory(sys, x_init: np.ndarray, steps: int, tol: float = 1e-8):
    x = np.asarray(x_init, dtype=float).copy()
    xs = [x.copy()]

    for _ in range(steps):
        idx = find_active_piece_index(sys.H_list, x, tol=tol)
        if idx is None:
            raise RuntimeError("No active piece found for current state (trajectory left all domains).")

        x = sys.A_list[idx] @ x
        xs.append(x.copy())

    return np.array(xs)


def plot_trajectory(ax, sys, x_init: np.ndarray, steps: int, tol: float = 1e-8):
    xs = simulate_trajectory(sys, x_init, steps=steps, tol=tol)
    x, y = xs[:, 0], xs[:, 1]
    ax.plot(x, y, "-o", markersize=2)
    return xs


# ---------- One-shot runner ----------

def plot_levelsets_and_trajectory(sys, H_list, c_list, steps: int = 100, level: float = 1e3, x_init=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect="equal")

    all_vertices = plot_all_piece_levelsets(ax, H_list, c_list, level=level)
    if x_init is None:
        if not all_vertices:
            raise RuntimeError("Could not find a feasible point to start the trajectory.")
        x_init = all_vertices[0]

    plot_trajectory(ax, sys, x_init, steps=steps)
    plt.show()
    return fig, ax
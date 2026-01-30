import gurobipy as gp
from gurobipy import GRB
import numpy as np

def linear_lyapunov_positive_system(
    A: np.ndarray,
    p_min: float = 1e-6,  # enforce p >= p_min
    verbose: bool = False
):
    """
    Compute a linear Lyapunov function V(x)=p^T x for a positive linear system.
    Discrete-time sufficient condition for x_{k+1} = A x_k:
        A^T p <= (1 - eps) p , p >= p_min, eps > 0
    We maximize eps.

    Returns
    -------
    p : np.ndarray
        Positive vector defining V(x)=p^T x
    eps : float
        Certified decay margin (larger is better). If eps <= 0, no strict certificate found.
    status : int
        Gurobi model status
    """

    A = np.asarray(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("A must be square.")
    norm_A = np.linalg.norm(A, np.inf)

    # Optional: quick positivity sanity check
    # Positive DT systems require A >= 0
    if np.any(A < -1e-12) and verbose:
        print("Warning: A has negative entries. Positivity assumptions may not hold.")

    model = gp.Model("linear_lyapunov_positive")
    model.Params.OutputFlag = 1 if verbose else 0

    # Decision variables
    p = model.addMVar(shape=n, lb=p_min, ub=1, name="p")   # p_min <= p <= 1
    eps = model.addVar(lb=-norm_A, name="eps")             # eps >= 0
    
    # Lyapunov inequalities (all componentwise)
    # A^T p <= (1 - eps) p
    # This is bilinear because eps * p is bilinear. To keep it linear, fix a scaling of p and use:
    # A^T p <= p - eps * 1 (a sufficient but stronger condition when p is normalized)
    model.addConstr(A.T @ p <= p - eps, name="decay_linear")

    # Objective: maximize eps (decay margin)
    model.setObjective(eps, GRB.MAXIMIZE)

    model.optimize()

    status = model.Status
    if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return None, None, status

    p_val = p.X
    eps_val = float(eps.X)
    return p_val, eps_val, status


if __name__ == "__main__":
    # Example: stable Positive matrix
    A = np.array([
        [0.0, 0.5, 0.0],
        [0.2, 0.5, 0.3],
        [0.0, 0.1, 0.0],
    ])
    print("spectral radius:", np.max(np.abs(np.linalg.eigvals(A))))

    p, eps, status = linear_lyapunov_positive_system(A, verbose=True)
    print("status:", status)
    print("p:", p)
    print("eps:", eps)
    if p is not None:
        print("Check A^T p:", A.T @ p - p)
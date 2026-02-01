import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import HalfspaceIntersection
from scipy.optimize import linprog
from src.system import PiecewiseLinearSystem, compute_path_graph
from src.stability import compute_lyapunov
from tests.utils import plot_levelsets_and_trajectory


# Stability
A_basic = 1.03 * np.array([[np.cos(1), -np.sin(1)], [np.sin(1), np.cos(1)]])
H0 = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
])
A0 = A_basic + 0.3 * np.array([[1, 1], [0, 0]])
H1 = np.array([
    [-1.0, 0.0],
    [0.0, 1.0],
])
A1 = A_basic + 0.3 * np.array([[-1, 1], [0, 0]])
H2 = np.array([
    [-1.0, 0.0],
    [0.0, -1.0],
])
A2 = A_basic + 0.3 * np.array([[-1, -1], [0, 0]])
H3 = np.array([
    [1.0, 0.0],
    [0.0, -1.0],
])
A3 = A_basic + 0.3 * np.array([[1, -1], [0, 0]])
sys = PiecewiseLinearSystem([H0, H1, H2, H3], [A0, A1, A2, A3])
node_list, edge_list = compute_path_graph(sys, 7)
H_list, c_list, eps, status = compute_lyapunov(sys, node_list, edge_list)
print("status:", status)
print("eps:", eps)

if eps is not None and eps > 1e-5:
    plot_levelsets_and_trajectory(sys, H_list, c_list)
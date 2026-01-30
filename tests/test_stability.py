import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import HalfspaceIntersection
from scipy.optimize import linprog
from src.system import PiecewiseLinearSystem, compute_path_graph
from src.stability import compute_lyapunov


print("=======================================================================")
print(" Stability")
print("=======================================================================")
H0 = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
])
A0 = np.array([
    [0.7, -0.7],
    [0.7, 0.7],
])
H1 = np.array([
    [-1.0, 0.0],
    [0.0, 1.0],
])
A1 = np.array([
    [0.7, -0.7],
    [0.7, 0.7],
])
H2 = np.array([
    [-1.0, 0.0],
    [0.0, -1.0],
])
A2 = np.array([
    [0.0, -1.0],
    [1.0, 0.0],
])
H3 = np.array([
    [1.0, 0.0],
    [0.0, -1.0],
])
A3 = np.array([
    [0.0, -0.5],
    [-0.5, -0.1],
])
sys = PiecewiseLinearSystem([H0, H1, H2, H3], [A0, A1, A2, A3])
node_list, edge_list = compute_path_graph(sys, 2)
print(node_list)
print(edge_list)
H_list, c_list, eps, status = compute_lyapunov(sys, node_list, edge_list)
print("status:", status)
print("eps:", eps)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, aspect='equal')

for (H, c) in zip(H_list, c_list):
    M = np.vstack((-H, c))
    b = np.vstack((np.zeros((H.shape[0], 1)), -1e3 * np.ones((1, 1))))
    halfspaces = np.hstack((M, b))
    norm_vector = np.reshape(np.linalg.norm(M, axis=1), (M.shape[0], 1))
    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A_ub = np.hstack((M, norm_vector))
    res = linprog(c, A_ub=A_ub, b_ub=-b, bounds=(None, None))
    feasible_point = res.x[:-1]
    rad = res.x[-1]
    if rad < 1e-8:
        continue
    hs = HalfspaceIntersection(halfspaces, feasible_point)
    vertices = [v for v in hs.intersections if np.linalg.norm(v) > 1e-6]
    vertices.insert(0, np.zeros(2))
    vertices.append(np.zeros(2))
    x, y = zip(*vertices)
    ax.plot(x, y, '-o', markersize=2)

plt.show()
print("=======================================================================")
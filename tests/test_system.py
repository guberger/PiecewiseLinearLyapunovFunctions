import numpy as np
from src.system import (PiecewiseLinearSystem, 
                        _check_path_feasibility,
                        _compute_feasible_path_list,
                        compute_path_graph)


def test_path_feasibility(sys, path):
    y, eps, status = _check_path_feasibility(sys, path, verbose=False)
    print("status:", status)
    print("y:", y)
    print("eps:", eps)
    if y is not None:
        print("Path feasible:", eps > 1e-8)


print("=======================================================================")
print(" System")
print("=======================================================================")
H0 = np.ones((3, 2))
A0 = np.ones((2, 2))
H1 = np.ones((3, 2))
A1 = np.ones((2, 2))
H2 = np.ones((3, 3))
A2 = np.ones((2, 2))
try:
    sys = PiecewiseLinearSystem([H0], [A0, A1])  # error
except ValueError:
    print("Error raised and caught!")
sys = PiecewiseLinearSystem([H0, H1], [A0, A1])  # no error
print("No error")
try:
    sys = PiecewiseLinearSystem([H0, H1, H2], [A0, A1, A2])  # error
except ValueError:
    print("Error raised and caught!")


print("=======================================================================")
print(" Path feasibility")
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
    [0.0, -1.0],
    [-1.0, 0.0],
])
sys = PiecewiseLinearSystem([H0, H1, H2, H3], [A0, A1, A2, A3])
print("---Length 2------------------------------------------------------------")
test_path_feasibility(sys, [0, 0])  # true
test_path_feasibility(sys, [0, 1])  # true
test_path_feasibility(sys, [0, 2])  # false
test_path_feasibility(sys, [0, 3])  # false
print("---Length 3------------------------------------------------------------")
test_path_feasibility(sys, [0, 0, 1])  # true
test_path_feasibility(sys, [0, 0, 2])  # false
test_path_feasibility(sys, [0, 1, 2])  # true
test_path_feasibility(sys, [0, 1, 3])  # false
print("---Length 4------------------------------------------------------------")
test_path_feasibility(sys, [0, 1, 2, 3])  # true
print("---Length 5------------------------------------------------------------")
test_path_feasibility(sys, [0, 1, 2, 3, 1])  # false
test_path_feasibility(sys, [0, 1, 2, 3, 2])  # false
print("---All paths: length 2-------------------------------------------------")
path_list = _compute_feasible_path_list(sys, 2)
print(path_list)
print("---All paths: length 3-------------------------------------------------")
path_list = _compute_feasible_path_list(sys, 3)
print(path_list)
print("---Path graph: length 2------------------------------------------------")
node_list, edge_list = compute_path_graph(sys, 2)
print(node_list)
print(edge_list)
print("=======================================================================")
import numpy as np
from src.system import (PiecewiseLinearSystem, 
                        _check_path_feasibility,
                        _compute_feasible_path_list,
                        compute_path_graph)


# PiecewiseLinearSystem
print("Start tests PiecewiseLinearSystem")

H0 = np.ones((3, 2))
A0 = np.ones((2, 2))
H1 = np.ones((3, 2))
A1 = np.ones((2, 2))
H2 = np.ones((3, 3))
A2 = np.ones((2, 2))

flag = False
try:
    sys = PiecewiseLinearSystem([H0], [A0, A1])
except ValueError:
    flag = True
assert flag
sys = PiecewiseLinearSystem([H0, H1], [A0, A1])
flag = False
try:
    sys = PiecewiseLinearSystem([H0, H1, H2], [A0, A1, A2])  # error
except ValueError:
    flag = True
assert flag

print("Tests ok!")


# Path feasibility
print("Start tests path feasibility")

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

# Length 2
assert _check_path_feasibility(sys, [0, 0])
assert _check_path_feasibility(sys, [0, 1])
assert not _check_path_feasibility(sys, [0, 2])
assert not _check_path_feasibility(sys, [0, 3])

# Length 3
assert _check_path_feasibility(sys, [0, 0, 1])
assert not _check_path_feasibility(sys, [0, 0, 2])
assert _check_path_feasibility(sys, [0, 1, 2])
assert not _check_path_feasibility(sys, [0, 1, 3])

# Length 4
assert _check_path_feasibility(sys, [0, 1, 2, 3])

# Length 5
assert not _check_path_feasibility(sys, [0, 1, 2, 3, 1])
assert not _check_path_feasibility(sys, [0, 1, 2, 3, 2])

# All paths: length 2
path_list = _compute_feasible_path_list(sys, 2)
# print(path_list)
assert len(path_list) == 10

# All paths: length 3
path_list = _compute_feasible_path_list(sys, 3)
# print(path_list)
assert len(path_list) == 20

# Path graph: length 2
node_list, edge_list = compute_path_graph(sys, 2)
# print(node_list)
# print(edge_list)
assert len(node_list) == 10
assert len(edge_list) == 20

print("Tests ok!")
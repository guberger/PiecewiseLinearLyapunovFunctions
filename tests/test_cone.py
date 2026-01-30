import numpy as np
from src.cone import _check_cone_nullity


def test_cone_nullity(H):
    y, eps, status = _check_cone_nullity(H, verbose=False)
    print("status:", status)
    print("y:", y)
    print("eps:", eps)
    if y is not None:
        print("Check y^T H:", y.T @ H)
        print("Equals zero:", eps < 1e-8)


print("=======================================================================")
print(" Cone nullity")
print("=======================================================================")
print("---Test 1--------------------------------------------------------------")
H = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 5.0],
    [0.5, -0.5, -0.5],
    [0.0, -1.0, 0.0],
])
test_cone_nullity(H)  # false
print("---Test 2--------------------------------------------------------------")
H = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 5.0],
    [0.5, -0.5, -0.5],
    [-1.0, 0.0, 0.0],
])
test_cone_nullity(H)  # true
print("---Test 3--------------------------------------------------------------")
H = np.array([
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, -1.0],
])
test_cone_nullity(H)  # false
print("=======================================================================")
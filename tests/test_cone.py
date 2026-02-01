import numpy as np
from src.cone import _check_cone_nullity


print("Start tests cone:")

# Test 1
H = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 5.0],
    [0.5, -0.5, -0.5],
    [0.0, -1.0, 0.0],
])
y, eps, status = _check_cone_nullity(H, verbose=False)
assert status == 2
assert y is not None
assert np.all(y >= 0.999)
assert eps > 1e-8


# Test 2
H = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 5.0],
    [0.5, -0.5, -0.5],
    [-1.0, 0.0, 0.0],
])
y, eps, status = _check_cone_nullity(H, verbose=False)
assert status == 2
assert y is not None
assert np.all(y >= 0.999)
assert eps < 1e-8
assert np.linalg.norm(y.T @ H) < 1e-8


# Test 3
H = np.array([
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, -1.0],
])
y, eps, status = _check_cone_nullity(H, verbose=False)
assert status == 2
assert y is not None
assert np.all(y >= 0.999)
assert eps > 1e-8

print("Tests ok!")
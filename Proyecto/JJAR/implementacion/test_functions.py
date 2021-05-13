"""
Pruebas unitarias para método región de confianza
"""

# Hack horrible por problemas de imports
import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import numpy as np
from region_confianza import region_confianza
from scipy.optimize import rosen


# Jacobiano analítico sacado a mano
def jaco(x):
    return np.array(
        [-400 * (x[1] - x[0] ** 2) * x[0] - 2 + 2 * x[0], 200 * x[1] - 200 * x[0] ** 2]
    )


# Hessiana analítica sacada a mano
def hesso(x):
    return np.array(
        [[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]]
    )


def test_region_confianza():
    x0 = np.array([5, 5])

    xk, pts = region_confianza(rosen, jaco, hesso, x0)

    assert np.isclose(xk, [1, 1], rtol=1e-9).all()

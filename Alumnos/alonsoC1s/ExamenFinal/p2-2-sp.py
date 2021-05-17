"""
Perdon no me da tiempo de hacer mi propia implementación porque no es un problema
trivial
"""

import numpy as np
import scipy.sparse.linalg as la
import scipy.sparse as sp
import random

random.seed(172255)

Diag_a = [random.randint(1, 1000) for x in range(1000000)]
b = [random.randint(1, 1000) for x in range(1000000)]

A = sp.diags(Diag_a)

# b no es sparse
b = np.array(b)

la.spsolve(A, b)
import numpy as np
from numpy import linalg as LA
from derivadas import Grad, cuadrados, rosenbrock, Hess
from wolfe import genera_alpha
from act_bfgs import BFGS_Hk, DFP_Hk


def BFGS(f, x0, tol, H0, maxiter=10000):
    k = 0
    Gk = Grad(f, x0)
    Hk = H0
    xk = np.array(x0)
    xk1 = np.array(x0)
    sk = np.array(100)

    while (LA.norm(Gk) > tol and LA.norm(sk) > tol and k <= maxiter):
        pk = - Hk.dot(Gk)
        alphak = genera_alpha(f, xk, pk)
        xk1 = xk + alphak * pk
        sk = xk1 - xk
        Gk1 = Grad(f, xk1)
        yk = Gk1 - Gk
        Hk = DFP_Hk(yk, sk, Hk)
        k += 1
        xk = xk1
        Gk = Gk1
    return xk1, k


if __name__ == "__main__":
    x, k = BFGS(rosenbrock, np.array([-100000, 1]), 1e-15, np.eye(2))
    print(f'Llegué a {x} en {k} iteraciones')
    x, k = BFGS(cuadrados, np.array([-10, 10, 20, 100]), 1e-15, np.eye(4))
    print(f'Llegué a {x} en {k} iteraciones')

import numpy as np
from numpy import linalg as LA
from derivadas import Grad, cuadrados
from wolfe import genera_alpha
from act_bfgs import  DFP_Hk


def BFGS_DFP(f, x0, tol, H0, maxiter=10000):
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
        "Actualización DFP"
        Hk = DFP_Hk(yk, sk, Hk)
        k += 1
        xk = xk1
        Gk = Gk1
    return xk1, k


if __name__ == "__main__":
    x, k = BFGS_DFP(cuadrados, [(-1)** i * 10 for i in range(10)], 1e-25, np.eye(10))
    print(f'Llegue a {x} en {k} iteraciones')

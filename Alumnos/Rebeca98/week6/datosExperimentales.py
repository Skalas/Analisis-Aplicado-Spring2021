import numpy as np
from derivadas import cuadrados

def condiciones_wolfe(f,x0,alpha,pk,tol = 1e-5):
    phi = lambda aplha : f(x0 + aplha*pk)

    print(phi(1))

    return phi

condiciones_wolfe(cuadrados, [1,1,1,1], 1, [-1,-1,-1,-1])

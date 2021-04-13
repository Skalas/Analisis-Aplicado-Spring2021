import numpy as np 
from derivadas import cuadrados

def condiciones_wolfe(f, x0, alpha, pk, tol=1e-5):
    phi = lambda alpha : f(x0 + alpha*pk)
    return phi(3)


if __name__ == '__main__':
    print(condiciones_wolfe(cuadrados, [1,1,1,1], 1, [-1,-1,-1,-1]))

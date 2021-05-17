import numpy as np
from numpy import linalg as LA

"""
Algoritmos usados
"""

def Grad(f, x0, h=1e-6, i=-1):
    n = len(x0)
    if i in range(n):
        z = np.zeros(n)
        z[i] = h/2
        Grad = (f(x0 + z) - f(x0 - z))/h
    else:
        Grad=np.zeros(n)
        for j in range(n):
            z = np.zeros(n)
            z[j] = h/2
            Grad[j]= (f(x0 + z) - f(x0 - z))/h
    return Grad

def genera_alpha(f, x0, pk, c1=1e-4, tol=1e-5):
    alpha, rho, c = 1, 4/5, c1
    while f(x0 + alpha*pk)>f(x0) + c*alpha*np.dot(Grad(f, x0),pk):
        alpha*=rho
    return alpha

def DFP_Hk(yk, sk, Hk):
    """
    Se suman las entradas ya que al realizar las multiplicaciones de vectores el resultado que debería ser de 1x1 queda
    como una mtariz diagonal
    """
    Hk1 = Hk - (Hk * yk * yk.T * Hk)/sum(sum(yk.T * Hk * yk)) + (sk * sk.T)/sum(yk.T * sk)
    return Hk1


"""
Definición de la función"
"""

def cuadrados(x):
    resultado = 0
    for i in range(len(x)):
        resultado += x[i]**2
    return resultado


"""
Definición de algoritmos
"""

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


"""
Implementación de la función
"""

x0=[(-1)**i*10 for i in range(10)]

x,k=BFGS(cuadrados,x0,1e-15,np.eye(10))




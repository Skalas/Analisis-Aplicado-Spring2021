import numpy as np
from numpy import linalg as LA

def Grad(f, x0, h=1e-6, i=-1):
    """
    Función que calcula el Grad de una función en un punto
    """
    n = len(x0)
    if i in range(n):
        z = np.zeros(n)
        z[i] = h/2
        Grad = (f(x0 + z) - f(x0 - z))/h
    else:
        Grad = np.zeros(n)
        for j in range(n):
            z = np.zeros(n)
            z[j] = h/2
            Grad[j] = (f(x0 + z) - f(x0 - z))/h
    return np.array(Grad)

def Hess(f, x0, h=1e-4, method="basic"):
    """
    Función que calcula la Hessiana  de una función en un punto.
    f: función sobre la cual queremos calcular la hessiana.
    x0: Punto sobre el cual queremos hacer el cálculo
    h: nivel de precisión para hacer el cálculo
    method: Método por el cual se quiere hacer puede ser:
             'basic', 'grad', 'centered', 'gradCentered'
    """
    n = len(x0)
    Hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            z_i = np.zeros(n)
            z_i[i] = h
            z_j = np.zeros(n)
            z_j[j] = h
            if method == "basic":
                Hess[i, j] = (f(x0 + z_j + z_i) - f(x0 + z_i) -
                              f(x0+z_j) + f(x0)) / (h**2)
            elif method == "grad":
                Hess[i, j] = (Grad(f, x0+z_j, h, i) - Grad(f, x0, h, i) +
                              Grad(f, x0+z_i, h, j) - Grad(f, x0, h, j))/(2*h)
            elif method == "centered":
                if i == j:
                    Hess[i, j] = (-f(x0+2*z_i) + 16*f(x0+z_i) - 30*f(x0) +
                                  16*f(x0-z_i) - f(x0-2*z_i)) / (12*h**2)
                else:
                    Hess[i, j] = (f(x0+z_i+z_j) - f(x0 + z_i - z_j) -
                                  f(x0 - z_i + z_j) + f(x0-z_i-z_j))/(4*h**2)
            elif method == "gradCentered":
                Hess[i, j] = (Grad(f, x0+z_j, h)[i] - Grad(f, x0-z_j, h)[i] +
                              Grad(f, x0+z_i, h)[j] - Grad(f, x0-z_i, h)[j])\
                               / (4 * h)
    return Hess

def cuadrados(x):
    resultado = 0
    for i in range(len(x)):
        resultado += x[i]**2
    return resultado

def DFP_Hk(yk, sk, Hk):
    """
    Función que calcula La actualización DFP de la matriz Hk
    In:
      yk: Vector n
      sk: Vector n
      Hk: Matriz nxn
    Out:
      Hk+1: Matriz nxn
    """

    yk = np.array([yk]).T
    sk = np.array([sk]).T


    Hk1 = Hk - (Hk * yk.dot(yk.T) * Hk)/(yk.T * Hk * yk) + (sk.dot(sk.T))/(yk.T.dot(sk))
    return Hk1

def genera_alpha(f, x0, pk, c1=1e-4, c2 = 0.5, tol=1e-5):
    """
    Backtracking LS i.e. Algoritmo que encuentra una
    alpha que cumpla condiciones de wolfe.
    """
    alpha, rho = 1, 3/4
    Gkpk = Grad(f, x0).dot(pk)
    while f(x0 + alpha*pk) > f(x0) + c1*alpha*Gkpk:
        alpha *= rho
    return alpha

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


x0 = [(-1)**i*10 for i in range(10)]
B0 = Hess(cuadrados, x0)
H0 = LA.inv(B0)
x, k = BFGS(cuadrados, x0, 1e-15, H0)
print(f' Llegué a {x} en {k} iteraciones')


    #Problema 2.2 

import random
random.seed(166088)
Diag_A = [random.randint(1,1000) for x in range(1000000)]
b = [random.randint(1,1000) for x in range(1000000)]
x0_2 = [(-1)**i*10 for i in range(1000000)]

def multi_rala(A,b):
    resultado = np.zeros(len(A))
    j = 0
    while j < len(A):
        resultado[j] = A[j]*b[j]
    return resultado

def gradiente_conjugado(x0, A, b):
    xk = x0
    b = np.array(b).T
    rk = np.zeros(len(A))
    j=0
    while j < len(A):
        rk[j] = A[j]*x0[j] - b[j]
        j = j+1
    pk = -rk
    while not (rk.T.dot(rk) ==  0):
        alphak = rk.T.dot(rk) / (pk.T.dot(multi_rala(A,pk)))
        alphak= alphak[0,0]
        xk_1 = xk + alphak.dot(pk)
        rk_1 =  rk + alphak.dot(multi_rala(A,pk))
        betak_1 = (rk_1.T.dot(rk_1)) / (rk.T.dot(rk))
        betak_1 = betak_1[0,0]
        pk_1 = -rk_1 + betak_1.dot(pk)
        xk, rk, pk = xk_1, rk_1, pk_1
    return xk

res = gradiente_conjugado(x0_2, Diag_A, b)
print(f'el resultado del sistema es {res}')
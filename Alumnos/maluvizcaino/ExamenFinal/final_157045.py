import random
import numpy as np
random.seed(157045)
Diag_A = [random.randint(1,1000) for x in range(1000000)]
b = [random.randint(1,1000) for x in range(1000000)]


def matriz_rala(n, Diag_A):
    rows, cols = (3, n)
    m_rala = [[0 for i in range(cols)] for j in range(rows)]
    k = 0
    for i in range(n):
        for j in range(n):
            if (Diag_A[i] != 0):
                m_rala[0][k] = i
                m_rala[1][k] = j
                m_rala[2][k] = Diag_A[i]
                k += 1
    for i in m_rala:
        print(i)
    return m_rala

matriz_rala(1000000, Diag_A)

def gradiente_conjugado(x0, A, b):
    xk = x0
    b = np.matrix(b).T
    #rk = np.dot(A, x0) - b
    #En lugar del producto punto
    for i in len(A):
        rk = np.sum(A[i] * x0[i]) - b
    pk = -rk
    while not (rk.T * rk ==  0):
        alphak = rk.T * rk / (pk.T * A * pk)
        alphak= alphak[0,0]
        xk_1 = xk + alphak * pk
        rk_1 =  rk + alphak * A * pk
        betak_1 = (rk_1.T * rk_1) / (rk.T * rk)
        betak_1 = betak_1[0,0]
        pk_1 = -rk_1 + betak_1 * pk
        xk, rk, pk = xk_1, rk_1, pk_1
    return xk


#No termine de correrlo porque me tomo mucho tiempo pensar en como realizar los productos de matrices con el arreglo de la matriz rala
print(gradiente_conjugado(x0, A, b, A))


#######################
#Pregunta 2.1
import numpy as np
import traceback
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

"""
Backtracking LS i.e. Algoritmo que encuentra una alpha que cumpla condiciones de wolfe.
"""
def genera_alpha(f, x0, pk, c1=1e-4, tol=1e-5):
    alpha, rho, c = 1, 4/5, c1
    while f(x0 + alpha*pk)>f(x0) + c*alpha*np.dot(Grad(f, x0),pk):
        alpha*=rho
    return alpha


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
    Hk1 = Hk + np.dot(sk, sk.T) / np.dot(sk.T, yk) - (np.dot(Hk, yk).dot(np.dot(yk.T, Hk))) / (np.dot(yk.T, Hk).dot(yk))
    return Hk1


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

def cuadrados(x):
    resultado = 0
    for i in range(len(x)):
        resultado += x[i]**2
    return resultado


x0 = [(-1)**i*10 for i in range(10) ]

x, k = BFGS(cuadrados, x0, 1e-15, np.eye(10))
print(f'Llegué a {x} en {k} iteraciones')

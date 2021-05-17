import numpy as np 
import random
from numpy import linalg as LA
from numpy.matrixlib.defmatrix import matrix


def cuadrados(x):
    resultado = 0
    for i in range(len(x)):
        resultado += x[i]**2
    return resultado

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




"""
a)
"""
def DFP_Hk(yk, sk, Hk):
    ym = np.dot(yk,yk.T)
    Hk1 = Hk - (np.dot(np.dot(Hk,ym),Hk))/(np.dot(yk.T,np.dot(Hk,yk))) + np.dot(sk,sk.T)/np.dot(yk.T,sk)
    return Hk1
    #El problema era dimensional

def DFP_Bk(yk,sk,Bk):
    """
    Función que calcula la actualización DFP de la matriz Bk
    yk : Vector n
    Sk : Vector n
    Bk : Matriz nxn
    Bk+1 : Matriz nxn
    """
    n = len(yk)
    rhok = 1/(yk.T*sk)
    Vk = (np.eye(n) - rhok * yk*sk.T)
    Bk1 = Vk * Bk * Vk.T + rhok * yk * yk.T
    return Bk1

def BFGS_Hk(yk, sk, Hk):
    """
    Función que calcula La actualización BFGS de la matriz Hk
    In:
      yk: Vector n
      sk: Vector n
      Hk: Matriz nxn
    Out:
      Hk+1: Matriz nxn
    """
    n = len(yk)
    yk = np.array([yk]).T
    sk = np.array([sk]).T
    rhok = 1 / yk.T.dot(sk)
    Vk = (np.eye(n) - rhok * yk.dot(sk.T))
    Hk1 = Vk.T * Hk * Vk + rhok * sk.dot(sk.T)
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
        Hk = DFP_Hk(yk, sk, Hk) #Aquí utilizamos la actualización DFP
        k += 1
        xk = xk1
        Gk = Gk1
    return xk1, k


"""
b
"""
random.seed(166780)
Diag_A = [random.randint(1,1000) for x in range(1000000)] #La matriz tiene ceros en las entradas distintas a la diagonal
b = [random.randint(1,1000) for x in range(1000000)]

def mult_rala(A,v, tol=1e-5): #multiplicación matriz rala A con v
    [n,m] = np.asarray(A.shape)
    Av=np.zeros(n,1)
    for i in range(n):
        j=1
        while j < m and A[i,j+1] < tol: #Con el tol nosotros decidimos que tan cercano a cero necesitamos el valor para considerarlo
            Av[i] = Av[i] + A[i,j] * v[A[i, j+1]]
            j += j+2
    return Av

def gradiente_conjugado(x0, A, b): #Nada más hay que cambiar como se multiplican las matrices, una vez definido en mult_rala
    xk = x0
    b = np.matrix(b).T
    rk = mult_rala(A,x0) - b
    pk = -rk
    while not(rk.T,rk == 0):
        alpha_k = rk.T*rk / (np.dot(pk.T,mult_rala(A,pk))) #Aquí va np.dot porque es multiplicación de vectores
        alpha_k = alpha_k[0,0]
        xk_1 = xk + alpha_k * pk     
        rk_1 = rk + alpha_k * mult_rala(A,pk) #alpha_k es una constante
        betak_1 = rk_1.T * rk_1 /(rk.T*rk)
        betak_1 = betak_1[0, 0]
        pk_1 = -rk_1 + betak_1 + pk 
        xk, rk, pk = xk_1, rk_1, pk_1
    return xk

#No pude poner la diagonal como si fuera matriz para poder hacer mult_rala y por eso no lo probé

if __name__ == "__main__":
    x0 = [(-1)**i*10 for i in range(10) ]
    sk = np.ones(1000)
    yk = np.ones(1000)
    H0 = np.eye(1000)
    #print(BFGS(cuadrados, x0, H0, maxiter=10000))



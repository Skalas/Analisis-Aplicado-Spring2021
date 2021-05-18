import numpy as np 
from numpy import linalg as LA
import random

from numpy.core.defchararray import not_equal

# Métodos preparativos
def gradiente(f, x0, h=1e-6, i=-1):
    """
    Función para calcular el gradiente de una función para un punto particular.
    """
    n = len(x0)
    if i in range(n):
        z = np.zeros(n)
        z[i]= h/2
        gradiente = (f(x0 + z) - f(x0 -z))/h
    else:
        gradiente = np.zeros(n)
        for j in range(n):
            z = np.zeros(n)
            z[j] = h/2
            gradiente[j] = (f(x0 + z) - f(x0 - z))/h
    return gradiente

def Hess(f, x0, h=1e-4, method="basic"):
    """
    Función para calcular la Hessiana de una función en un punto particular.
    f: función a la cual se le calculará la Hessiana
    x0: Punto sobre el cual se hará el cálculo
    h: nivel de precisión o tolerancia
    method: Método por el cual se calcula (basic, grad, centered o gradCentered)
    """
    n = len(x0)
    Hess = np.matrix(np.zeros((n,n)))
    for i in range(n):
        for j in range(n):
            z_i = np.zeros(n)
            z_j = np.zeros(n)
            if j<= i:
                z_i[i] = h
                z_j[j] = h
                if method == "basic":
                    Hess[i,j] = (f(x0 + z_j + z_i) - f(x0 + z_i) - f(x0 + z_j) + f(x0))/ (h**2)
                    Hess[j,i] = Hess[i,j]
                elif method == "grad":
                    Hess[i,j] = (gradiente(f, x0+z_j, h, i) - gradiente(f, x0, h, i) + gradiente(f, x0+z_i, h, j) - gradiente(f, x0, h, j))/(2*h)
                    Hess[j,i] = Hess[i,j]
                elif method == "centered":
                    if i==j:
                        Hess[i,j] = (-f(x0+2*z_i) + 16*f(x0+z_i) - 30*f(x0) + 16*f(x0-z_i) - f(x0-2*z_i))/(12*h**2)
                        Hess[j,i] = Hess[i,j]
                    else:
                        Hess[i,j] = (f(x0+z_i+z_j) - f(x0+z_i-z_j) - f(x0-z_i+z_j) + f(x0-z_j-z_j))/(4*h**2)
                        Hess[j,i] = Hess[i,j]
                elif method == "gradCentered":
                    Hess[i,j] = (gradiente(f, x0+z_j,h)[i] - gradiente(f, x0-z_j, h)[i] + gradiente(f, x0+z_i,h)[j] - gradiente(f, x0-z_i,h)[j])/(4*h)
                    Hess[j,i] = Hess[i,j]
    return Hess

def genera_alpha(f, x0, pk, c1=1e-4, tol=1e-15):
    """
    Backtracking LS 
    (algoritmo para encontrar el alpha que cumpla las condiciones de Wolfe)
    """
    alpha, rho = 1, 3/4
    Gkpk = gradiente(f, x0).dot(pk)
    while f(x0 + alpha*pk) > f(x0) + c1*alpha*Gkpk:
        alpha *= rho
    return alpha

"""
----------------------------------------
Ejercicio 2.1 DFP
----------------------------------------
"""

#Instanciación de funcion

def cuadrados(x):
    resultado = 0
    for i in range(len(x)):
        resultado += x[i]**2

    return resultado

# Métodos de actualización DFP

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
    Hk1 = Hk - (np.dot(Hk, np.dot(yk, np.dot(yk, Hk))))/(np.dot(yk, np.dot(Hk,yk))) + np.dot(sk, sk)/np.dot(yk,sk)
    return Hk1

def DFP_Bk(yk, sk, Bk):
    """
    Función que calcula La actualización DFP de la matriz Hk
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
    rhok = 1 / (yk.T.dot(sk))
    Bk1 = ((np.eye(n) - rhok*yk.T*sk).T)*Bk*(np.eye(n) - rhok*yk.T*sk) + rhok*yk*yk.T
    return Bk1

# Método BFGS 

def DFP1(f, x0, tol, H0, maxiter=10000):
    """
    Método BFGS utilizando la actualización DFP de Hk
    """
    k = 0
    Gk = gradiente(f, x0)
    Hk = H0
    xk = np.array(x0)
    xk1 = np.array(x0)
    sk = np.array(100)
    while (LA.norm(Gk) > tol and LA.norm(sk) > tol and k <= maxiter):
        pk = - Hk.dot(Gk)
        alphak = genera_alpha(f, xk, pk)
        xk1 = xk + alphak * pk
        sk = xk1 - xk
        Gk1 = gradiente(f, xk1)
        yk = Gk1 - Gk
        Hk = DFP_Hk(yk, sk, Hk)
        k += 1
        xk = xk1
        Gk = Gk1
    return xk1, k

def DFP2(f, x0, tol, B0, maxiter=10000):
    """
    Método BFGS utilizando la actualización DFP de Hk
    """
    k = 0
    Gk = gradiente(f, x0)
    Bk = B0
    xk = np.array(x0)
    xk1 = np.array(x0)
    sk = np.array(100)
    while (LA.norm(Gk) > tol and LA.norm(sk) > tol and k <= maxiter):
        pk = - Bk.dot(Gk)
        alphak = genera_alpha(f, xk, pk)
        xk1 = xk + alphak * pk
        sk = xk1 - xk
        Gk1 = gradiente(f, xk1)
        yk = Gk1 - Gk
        Bk = DFP_Bk(yk, sk, Bk)
        k += 1
        xk = xk1
        Gk = Gk1
    return xk1, k

# Prueba
n = 10
x0 = [(-1)**i*10 for i in range(n)]
x, k = DFP1(cuadrados, np.array(x0), 1e-15, np.eye(n))
print(f'Llegué a {x} en {k} iteraciones')

x, k = DFP2(cuadrados, np.array(x0), 1e-15, np.eye(n))
print(f'Llegué a {x} en {k} iteraciones')


"""
----------------------------------------
Ejercicio 2.2 Gradiente Conjugado
----------------------------------------
"""
#Instanciación de sistema lineal Ax=b

random.seed(174308)
Diag_A = [random.randint(1,1000) for x in range(1000000)]
b = [random.randint(1,1000) for x in range(1000000)]

# Implementación matriz rala
"""
#Guarda el número que es distinto de cero y la posición de la matriz (como es diagonal, i=j)
"""
n=2
A_rala = [[0] * n for i in range(10**6)]

for i in range(len(Diag_A)):
    if Diag_A[i] != 0:
        A_rala[i][0] = Diag_A[i]
        A_rala[i][1] = i

# Método Gradiente Conjuado

def gradienteConjugado(x0, A, b):
    xk = x0
    b = np.matrix(b).T
    rk = np.dot(A, x0) - b
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

def gradienteConjugado_Precond(x0, A, b, M):
    xk = x0
    b = np.matrix(b).T
    rk = np.dot(A, x0) - b
    yk = np.linalg.solve(M, rk)
    pk = -yk 
    while not (rk.T * rk ==  0):
        alphak = rk.T * yk / (pk.T * A * pk)
        alphak= alphak[0,0]
        xk_1 = xk + alphak * pk
        rk_1 =  rk + alphak * A * pk
        yk_1 = np.linalg.solve(M, rk_1)
        betak_1 = (rk_1.T * rk_1) / (rk.T * rk)
        betak_1 = betak_1[0,0]
        pk_1 = -yk_1 + betak_1 * pk
        xk, rk, pk, yk  = xk_1, rk_1, pk_1, yk_1
    return xk

# Pruebas
#print(gradienteConjugado(x0, A_rala, b))
#print(gradienteConjugado_Precond(x0, A_rala, b, A_rala)) 
"""
No pude terminar de modificar el metodo para que acepte la matriz rala
"""

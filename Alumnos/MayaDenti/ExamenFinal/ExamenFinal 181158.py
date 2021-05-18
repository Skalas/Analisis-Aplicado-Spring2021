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

def genera_alpha(f, x0, pk, c1=1e-4, c2 = 0.5, tol=1e-5):
    """
Backtracking LS i.e. Algoritmo que encuentra una
alpha que cumpla condiciones de wolfe.
    """
    alpha, rho = 1, 3/4
    Gk=np.matrix(Grad(f, x0))
    Gkpk = np.dot(Gk,pk)
    while f(x0 + alpha*pk) > f(x0) + c1*alpha*Gkpk:
        alpha *= rho
    return alpha

"""
Ejercicio 2.1
"""

def DFP_Hk(yk, sk, Hk):
    """
Función que calcula La actualización DFP de la matriz Hk
    IN:
      yk: Vector n
      sk: Vector n
      Hk: Matriz nxn
    OUT:
      Hk+1: Matriz nxn
    """
    Hk1 = Hk - (Hk * yk * yk.T * Hk)/(yk.T * Hk * yk) + (sk * sk.T)/(yk.T * sk)
    return Hk1


def BFGS(f, x0, tol, H0, maxiter=10000):
    """
Función que minimiza f con BFGS y modificación Hk
    IN
        f: función a minimizar
        x0: punto inicial
        tol: tolerancia
        H0: matriz pos def y simétrica (puede ser un multiplo de I)
    OUT:
        xk: punto mínimo
    """
    k = 0
    Gk = Grad(f, x0)
    Hk = H0
    xk = np.array(x0)
    sk = np.array(100)
    while (LA.norm(Gk) > tol and LA.norm(sk) > tol and k <= maxiter):
        
        pk = - Hk.dot(Gk) 
        #Había un problema en el genera alpha por la multiplicación con pk entonces aquí lo arreglo haciendolo un array
        if type(pk) == np.matrix:
            pk = pk.tolist()
            pk = np.array(pk[0])
        
        alphak = genera_alpha(f, xk, pk)
        xk1 = xk + alphak * pk
        
        sk = xk1 - xk
        sk=np.matrix(sk).T #La transforme a matriz para que funcione la multiplicación en el DFP_Hk
       
        Gk1 = Grad(f, xk1)
        
        yk = Gk1 - Gk
        yk= np.matrix(yk).T #La transforme a matriz para que funcione la multiplicación en el DFP_Hk
        
        Hk = DFP_Hk(yk, sk, Hk) #Cambie aquí el BFGS_Hk por DFP_Hk pues es lo que nos pide el ejercicio
        
        k += 1
        xk = xk1
        Gk = Gk1
    return xk1, k

def cuadrados(x):
    resultado = 0
    for i in range(len(x)):
        resultado += x[i]**2
    return resultado

"""
Resultados 2.1
"""

x, k = BFGS(cuadrados, np.array([(-1)**i*10 for i in range(10)]), 1e-15, np.eye(10))
print("El punto inicial es:",np.array([(-1)**i*10 for i in range(10)])) # combrobe el punto final con BFGS_Hk y es aproximadamente 0, igual que este
print(f'Llegué a {x} en {k} iteraciones')

#Vemos que converge a un vector casi de ceros (que es el mínimo de una función cuadrada en dimensión 10)

"""
Ejercicio 2.2
"""

import random

random.seed(181158) #  Cambien a su propia clave
Diag_A = [random.randint(1,1000) for x in range(1000000)]
b = [random.randint(1,1000) for x in range(1000000)]
"""
Lo que quiero es solo resolver el sistema ignorando todas las entradas 0 entonces es 
sólo multiplicar las entradas distintas de 0 en la diagonal de A por la entrada correspondiente del vector x igualada
a la entrada correspondiente del vector b
"""

A=[] #creo el vector con entradas distintas de 0
for i in range(1000000):
    if Diag_A[i]!=0:
        A.append(Diag_A[i]) #creo un vector con las entradas de la diagonal de A distintas a 0
print(len(A)) #quería ver si se achicaba el vector pero me queda del mimso tamaño entonces no sé bien qué esta saliendo mal

def gradiente_conjugado_precond(x0, A, b, M):
    """
Algoritmo de gradiente conjugado precondicionado que minimiza
    phi(x) = (1/2) x^T*A*x - b^T*x
    IN:
        x0: punto inicial
        A: matriz simétrica positva definida nxn
        b: vector de dimensión n
        M: matriz nxn (M=C^T*C, C matriz no singular tal que x'=Cx)
    OUT:
        xk: vector de dimensión n (el mínimo)
    """
    xk = x0
    A=np.matrix(A).T # Lo convierto en matriz para que no haya problema en las muliplicaciones
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

"""
Resultados 2.2
"""
#Si corro este segundo ejercicio no se rompe el programa pero durante el tiempo de examen no convergió a nada

"""
x0=np.zeros(1000000)
M = np.matrix(np.eye(1000000)).T
#print(gradiente_conjugado_precond(x0, A, b, M))
"""
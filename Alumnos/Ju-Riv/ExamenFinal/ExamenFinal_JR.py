"""
Análisis Aplicado
Examen Final
Julieta Rivero
"""

import numpy as np
from numpy import linalg as la
from LabClase5 import Grad
from LabClase7 import genera_alpha

def cuadrados(x, n=2):
    """
    Eleva a la n-ésima potencia todas las entradas del vector x dado
    y suma las entradas de este nuevo vector
    potencias(x,n) = sum(xi^n)
    """        
    resultado = 0
    for i in range(len(x)):
        resultado += x[i]**n
    return resultado

def DFP_Hk(yk, sk, Hk):
    """
    Función que calcula La actualización DFP de la matriz Hk
    
    IN
      yk: vector dimensión n
      sk: vector dimensión n
      Bk: matriz nxn
      
    OUT
      Bk+1: matriz nxn
    """
    Hk_1 = Hk - (Hk * yk * yk.T * Hk)/(yk.T * Hk * yk)
    Hk_1 = Hk_1 + (sk * sk.T)/(yk.T * sk)
    return Hk_1

def BFGS(f, x0, tol, H0, maxiter=200):
    """
    Función que implementa el algoritmo BFGS con la actualización
    DFP para Hk
    
    IN
        f: función a optimizar
        xo: punto inicial, vector dimesión n
        tol: tolerancia
        H0: aproximaxión de la Hessiana
        maxiter: cota superior para las iteraciones
        
    OUT: 
        xk_1: punto óptimo, vector de dimesión n
        k: iteraciones totales
        
    """
    k = 0
    Gk = Grad(f, x0)
    Hk = H0
    xk = x0
    sk = np.array(100)
    while (la.norm(Gk) > tol and la.norm(sk) > tol and k <= maxiter):
        pk = - Hk.dot(Gk)
        if type(pk) == np.matrix:
            pk = pk.tolist()
            pk = np.array(pk[0])
            
        alphak = genera_alpha(f, xk, pk)
        xk_1 = xk + alphak * pk

        sk = xk_1 - xk
        sk = np.matrix(sk).T

        Gk_1 = Grad(f, xk_1)
        yk = Gk_1 - Gk
        yk = np.matrix(yk).T

        Hk = DFP_Hk(yk, sk, Hk)
        Gk = Gk_1

        k += 1
        xk = xk_1
        # print('xk: ', xk, '\nalpha_k: ', alphak, '\npk: ', pk, '\nk: ', k)
        # print('\n')
        
    return xk_1, k

if __name__ == "__main__":
    x0 = np.array([(-1)**i*10 for i in range(10)])
    x, k = BFGS(cuadrados, x0, 1e-15, np.eye(10))
    print(f'Llegué a {x} en {k} iteraciones')
    
#------------------------------------------------
import random
        
random.seed(170454)
Diag_A = [random.randint(1,1000) for x in range(1000000)]
b = [random.randint(1,1000) for x in range(1000000)]
    
def mat_diag(diag):
    """
    Función que genera un arreglo que en cada entrada tiene a (i, Aii), 
    donde Aii es el valor de la diagonal de A por renglón/columna e i
    representa el índice del renglón/columna
    
    IN
        diag: vector con las entradas de la diagonal
    
    OUT
        A: representación de la matriz diagonal A
    """
    A = []
    for i in range(len(diag)):
        A.append([i, diag[i]])
    A = np.array(A)
    return A

if __name__ == "__main__":
    A = mat_diag(Diag_A)
    print(A[571])

def dot_mdiag(m_diag, x0):
    """
    Función que calcula el producto de una matrix diagonal por un vector x0
    """
    dot_mx = []
    for i in range(len(m_diag)):
        dot_mx.append([m_diag[i,1]*b[i]])
    dot_mx = np.array(dot_mx)
    return dot_mx

if __name__ == "__main__":
    x0 = np.array([1 for i in range(1000000)])
    Ax = dot_mdiag(A, x0)

def gradiente_conjugado(x0, m_diag, b, maxiter = 200):
    """
    Esta función implementa la versión final del algoritmo de gradiente
    conjugado que minimiza a la función para A matriz diagonal
                    phi(x) = (1/2) x^T*A*x - b^T*x
    
    IN 
        xo: punto inicial 
        m_diag: matriz nxn diagonal
        b: vector de dimensión n
    
    OUT
        xk.T: vector de dimensión n
    """
    xk = x0
    Ax = dot_mdiag(m_diag, x0)
    rk = Ax.T - b
    pk = -rk
    k = 0
    
    while (not np.dot(rk, rk.T) == 0) and k <= maxiter:
        Ap = dot_mdiag(m_diag, pk.T)
        pAp = dot_mdiag(pk, Ap)
        alpha_k = np.dot(rk, rk.T) / pAp
        
        xk_1 = xk.T + alpha_k * pk
        rk_1 = rk + (alpha_k[0,0] * Ap.T)
        Bk_1 = np.dot(rk_1, rk_1.T) / np.dot(rk, rk.T)
        pk_1 = -rk_1 + Bk_1*pk
        
        xk, rk, pk, k = xk_1, rk_1, pk_1 , k+1
    
    return xk, k

if __name__ == '__main__': 
    print(gradiente_conjugado(x0, A, b))
    
# Ya no me dió tiempo de corregir gradiente_conjugado :( 

# Aunque calculé los productos puntos por separado obtengo el error: 
# Unable to allocate 7.28 TiB for an array with shape (1000000, 1000000)

# Creo que tendría que meterle if's al algoritmo para calcular
# alphak, rk_1 entrada por entrada y luego reconstruir los vectores
# xk_1, Bk_1, pk_1 en cada iteración
# -*- coding: utf-8 -*-
"""
Análisis Aplicado
Clase 01-03

"""
import numpy as np
from LabClase5 import Grad
from LabClase3 import potencias

def Condiciones_Wolfe(f, x0, alpha, pk, c1 = 1e-4, c2 = 1e-2, tol = 1e-5):
    """
    Determina si la función f dada, evaluada en x0, cumple las 
    Condiciones de Wolfe
    """
    phi = lambda alpha : f(x0 + alpha*pk)
    # Recordar: <x,y> = a * b^T
    linea = lambda alpha : f(x0) + c1 * alpha * np.dot(Grad(f, x0), pk)
    cond_1 = linea(alpha) - phi(alpha)
    cond_2 = np.dot(Grad(f, x0 + alpha * pk), pk) 
    cond_2 += - c2 * np.dot(Grad(f, x0), pk)
    return cond_1 >= 0 and cond_2 >= 0

def genera_alpha(f, x0, pk, c1 = 1e-4, tol = 1e-5):
    """
    Backtraking Line Search
    i.e. Algortimo que encuentra una alpha que cumple con las condiciones de 
         Wolfe
    """
    alpha = 1
    rho = 3/4
    while f(x0 + alpha * pk) > f(x0) + c1 * alpha * np.dot(Grad(f, x0), pk):
        alpha = alpha * rho
    return alpha

x = np.array([1,1,1,1])
pk = np.array([-1,1,-1,-1])


if __name__ == '__main__':
    alpha = genera_alpha(potencias, x, pk)
    print(alpha)
    print(Condiciones_Wolfe(potencias, x, alpha, pk))
    
def is_pos_def(Hessiana, tol = 1e-12):
    if np.all(np.linalg.eigvals(Hessiana) > tol):
            return True
    else:
            return False

def modificacion_hessiana(Hessiana, lam = 0.5):
    while not is_pos_def(Hessiana):
        Hess_mod = Hessiana + lam * np.eye(len(Hessiana))
    return Hess_mod


    
    


    
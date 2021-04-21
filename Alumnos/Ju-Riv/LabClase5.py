# -*- coding: utf-8 -*-
"""
Análisis Aplicado
Clase 17-02

"""
import numpy as np
from LabClase3 import potencias

def Grad(f, x0, h = 1e-6, i = -1):
    """
    Función que calcula el gradiente de una función f en un punto x0
    Para i dada la función regresa la i-ésima entrada del vector gradiente, 
    de lo contrario la función regresa todo el vector
    """
    n = len(x0)
    if i in range(n):
        z = np.zeros(n)
        z[i] = h/2
        Grad = (f(x0 + z) - f(x0 - z))/h
    else:
        Grad = np.zeros(n)
        for k in range(n):
            z = np.zeros(n)
            z[k] = h/2
            Grad[k]= (f(x0 + z) - f(x0 - z))/h
    return Grad

# Grad(f, x0, h, i) == Grad(f, x0, h)[i]
x = [1,1,1,1,1,1,1]
print(Grad(potencias, x, h = 0.000001))

def Hessiana(f, x0, h = 1e-4, method = "basic"):
    """
    Esta función regresa una matriz con la Hessiana de f evaluada en x0
    El parámetro method determina cómo se calcula la Hessiana
    https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm
    """  
    n = len(x0)
    Hess = np.matrix(np.zeros((n,n)))
    for i in range(n):
        for j in range(n):
            z_i = np.zeros(n)
            z_j = np.zeros(n)
            z_i[i] = h
            z_j[j] = h
            if method == "basic":
                Hess[i,j] = ( f(x0 + z_j + z_i) - f(x0 + z_i ) +\
                             - f(x0 + z_j) + f(x0)) / (h**2)
            elif method == "grad":
                Hess[i,j] = (Grad(f, x0 + z_j, h, i) - Grad(f, x0, h, i) + \
                             Grad(f, x0 + z_i, h, j) - Grad(f, x0, h,j ))/(2*h)
            elif method == "centered":
                if i == j:
                    Hess[i,j] = (-f( x0 + 2*z_i) + 16*f(x0 + z_i) - 30*f(x0)+\
                                 16*f(x0 - z_i) - f(x0 - 2*z_i))  / (12*h**2)
                else:
                    Hess[i,j] = (f(x0 + z_i + z_j) - f(x0 + z_i - z_j) - \
                                 f(x0 - z_i + z_j) + f(x0 -z_i - z_j))/(4*h**2)
            elif method == "gradCentered": 
                Hess[i,j] = (Grad(f, x0 + z_j, h)[i] - Grad(f, x0 - z_j, h)[i] + \
                                 Grad(f, x0 + z_i, h)[j] - Grad(f, x0 - z_i, h)[j])/(4*h)
    return Hess


x = [0,0]
print(potencias(x))
print(Hessiana(potencias, x, h = 1e-5))


# -*- coding: utf-8 -*-
"""
Análisis Aplicado
Clase 25-01

"""
import numpy as np

def gradiente(f, x0, h = 1e-12):
    """
    Calcula el gradiente de una función f en un punto dado x0
    h = e^(-12) ~ 6.14 x 10^(-6) 
    """
    n = len(x0)
    grad = np.zeros(n)
    
    for i in range(n):
        z = np.zeros(n)
        z[i] = h/2
        grad[i] = (f(x0 + z) - f(x0 - z))/h
        
    return grad

def potencias(x, n=2):
    """
    Eleva a la n-ésima potencia todas las entradas del vector x dado
    y suma las entradas de este nuevo vector
    potencias(x,n) = sum(xi^n)
    """        
    resultado = 0
    for i in range(len(x)):
        resultado += x[i]**n
    return resultado

x = np.array([1,1,1,1])
print(type(x))
print(potencias(x))

# gradiente(potencias, x) = (sum(xi^n))' = n(sum(xi^(n-1))) 
print(gradiente(potencias, x, h = 0.000001))





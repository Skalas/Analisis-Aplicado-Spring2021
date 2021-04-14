import numpy as np
import math
from Clase_15_Febrero import CondPrimerOrden

"""
Buscamos modificar nuestra función original gradiente(25 de enero) para que nuestra función Hessiana(17 de febrero),
en method == 'grad', pueda reescribirse gradiente(f,x0+z_j,h,i) en lugar de gradiente(f,x0+z_j,h)[i]
"""

def Grad(f,x0,h=1e-6, i=-1): #función gradiente renovada
    n = len(x0)
    if i in range(n):
        z = np.zeros(n)
        z[i] = h/2
        grad = (f(x0+z)-f(x0-z))/h
    else:
        grad = np.zeros(n)
        for j in range(n):
            z = np.zeros(n)
            z[j] = h/2
            grad[j] = (f(x0+z)-f(x0-z))/h
    return grad

def Hessiana(f,x0,h=1e-4, method = "basic"):
    n =  len(x0)
    Hess = np.matrix(np.zeros((n,n)))
    for i in range(n):
        for j in range(n):
            z_i = np.zeros(n)
            z_i[i] = h
            z_j = np.zeros(n)
            z_j[j] = h
            if method == "basic":
                Hess[i,j] = ( f(x0 + z_j + z_i) - f(x0 + z_i) - f(x0 + z_j) + f(x0))/(h**2)
            elif method == "grad":
                Hess[i,j] = (Grad(f,x0+z_j,h,i) - Grad(f,x0,h,i) + Grad(f,x0+z_i,h,j) - Grad(f,x0,h,j))/(2*h)
            elif method == "centered":
                if i==j:
                    Hess[i,j] = (-f(x0+2*z_i) + 16*f(x0+z_i) - 30*f(x0) + 16*f(x0-z_i) - f(x0-2*z_i))/(12*h**2)
                else:
                    Hess[i,j] = (f(x0+z_i+z_j) - f(x0 + z_i - z_j) - f(x0 - z_i + z_j)+ f(x0-z_i-z_j))/(4*h**2)
            elif method == "gradCentered":
                Hess[i,j] = (Grad(f,x0+z_j,h)[i] - Grad(f,x0-z_j,h)[i] +\
                     Grad(f,x0+z_i,h)[j] - Grad(f,x0-z_i,h)[j])/(4*h)
    return Hess

def cuadrado(x):
    resultado=0
    for i in range(len(x)):
        resultado+= x[i]**4
    return resultado

#print(Grad(cuadrado,[0,0,0,0],h=0.0000001))
#print(Hessiana(cuadrado, [0,0,0,0],h=1e-5,method="grad"))




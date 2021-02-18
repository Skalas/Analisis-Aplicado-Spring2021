import numpy as np

def gradiente(f, x0, h=1e-12):
    """
    Función que calcula el gradiente de una función en un punto
    """
    n = len(x0)
    Grad=np.zeros(n)
    for i in range(n):
        z = np.zeros(n)
        z[i] = h/2
        Grad[i]= (f(x0 + z) - f(x0 - z))/h
    return Grad

def cuadrados(x):
    resultado=0
    for i in range(len(x)):
        resultado+=x[i]**2
    return resultado

print(gradiente(cuadrados,[1,1,1,1],h=0.0000001)) 

    

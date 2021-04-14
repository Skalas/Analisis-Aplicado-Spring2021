import numpy as np

def gradiente(f, x0, h=0.0001):
    n = len(x0)
    Grad=np.zeros(n)
    for i in range(n):
        z=np.zeros(n)
        z[i] = h
        Grad[i]= (f(x0 +z) - f(x0))/h
        return Grad

def cuadrados(x):
    resultado=0
    for i in range(len(x)):
        resultado+=x[i]**2
    return resultado

print(gradiente(cuadrados,[1,1,1,1],h=0.0000001)) 


#Función del gradiente
import math
import numpy as np

def gradiente(f,x0,h=0.0001):
    n = len(x0)
    grad = np.zeros(n)
    for i in range(n):
        z = np.zeros(n)
        z[i] = h/2 #antes de la clase del 17 de febrero aquí iba z[i] = h
        grad[i] = (f(x0 + z) - f(x0))/h #y aquí estaba dividido entre 2*h
    return grad

def cuadrados(x): #f, en este caso solo le estamos dando la función como objeto
    resultado = 0
    for i in range(len(x)):
        resultado += x[i]**2
    return resultado

#print(gradiente(cuadrados,[1,1,1,1])) 




import numpy as np
import math
from Clase_25_Enero import gradiente, cuadrados
from Clase_15_Febrero import CondPrimerOrden

#Vamos a determinar la condición faltante, la de segundo orden 

def Hessiana(f,x0,h=0.0001, method = "basic"):
    """
    Función que calcula la Hessiana de uan función en un punto.
    f: función sobre la cual queremos calcular la hessiana 
    x0: Punto sobre el cual queremos hacer el cálculo
    h: nivel de presición para hacer el cálculo 
    method: Método por el cual se quiere hacer puede ser: 'basic', 'grad', 'centered', 'gradCentered'
    """
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
                Hess[i,j] = (gradiente(f,x0+z_j)[i] - gradiente(f,x0,h)[i] +\
                    gradiente(f,x0+z_i,h)[j] - gradiente(f,x0,h)[j]/(2*h))
            elif method == "centered":
                if i==j:
                    Hess[i,j] = (-f(x0+2*z_i) + 16*f(x0+z_i) - 30*f(x0) + 16*f(x0-z_i) - f(x0-2*z_i))/(12*h**2)
                else:
                    Hess[i,j] = (f(x0+z_i+z_j) - f(x0 + z_i - z_j) - f(x0 - z_i + z_j)+ f(x0-z_i-z_j))/(4*h**2)
            elif method == "gradCentered":
                Hess[i,j] = (gradiente(f,x0+z_j,h)[i] - gradiente(f,x0-z_j,h)[i] +\
                     gradiente(f,x0+z_i,h)[j] - gradiente(f,x0-z_i,h)[j])/(4*h)
    return Hess
 

#Para jugar con Python en la terminal le doy al szh python y para salir exit()
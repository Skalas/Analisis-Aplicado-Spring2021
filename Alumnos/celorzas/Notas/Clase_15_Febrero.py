import math
import numpy as np 
from Clase_25_Enero import gradiente,cuadrados

#Vamos a determinar dos condiciones: La primera, si nuestra derivada de primer orden(gradiente) es cero 
#Generamos una función que nos regrese verdadero o falso, si el punto x0 es crítico
def CondPrimerOrden(f,x0,tolerancia):
    grad=gradiente(f,x0)
    normaCuadrado=np.dot(grad,grad)
    if normaCuadrado<=tolerancia:
        return True
    else:
        return False

#print(CondPrimerOrden(cuadrados,[0,0,0,0],0.0001))
#La tolerancia de las dos funciones se deben de llevar, en gradiente pedíamos 0.0001 


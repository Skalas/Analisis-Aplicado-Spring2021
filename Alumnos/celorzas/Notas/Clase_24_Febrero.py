import numpy as np 
import math 
from Clase_22_Febrero import Grad,Hessiana 
from Clase_15_Febrero import CondPrimerOrden
from Clase_25_Enero import cuadrados

def CondSegOrden(f,x0,tol=1e-5):
    """
    Código de segundo orden
    """
    hess = Hessiana(f,x0)
    if np.all(np.linalg.eigvals(hess) > tol) : 
        return True
    else:
        return False

#print(Hessiana(cuadrados, [0,0,0,0],h=1e-5,method="grad"))
#print(CondSegOrden(cuadrados, [0,0,0,0]))

def is_min(f,x0,tol=0.0001):
    """
    Dado una función en un punto, la función is_min nos va a regresar True si es un mínimo, False si
    no podemos garantizar que sea un mínimo 
    """
    if CondPrimerOrden(f,x0,tol) and CondSegOrden(f,x0):
        return True
    else:
        return False
    
#print(is_min(cuadrados,[1,0,0,0]))
#print(CondPrimerOrden(cuadrados,[1,0,0,0],0.0001))
#print(CondSegOrden(cuadrados,[1,0,0,0]))




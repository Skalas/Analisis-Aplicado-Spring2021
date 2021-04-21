# -*- coding: utf-8 -*-
"""
Análisis Aplicado
Clase 22-02

"""
import numpy as np
from LabClase3 import potencias
from LabClase4 import First_Order_Conditions
from LabClase5 import Hessiana
    
def Second_Order_Conditions(f, x0, tol = 1e-5):
        """
        Evalua si la matriz Hessiana de f evaluada en x0 es positiva definida
        Recordar: una matriz es positiva definida si la parte real de todos 
                  sus eigenvalores es positiva
        """
        Hess = Hessiana(f, x0, tol)
        if np.all(np.linalg.eigvals(Hess) > tol):
            return True
        else:
            return False

x = [0,0]
print(Second_Order_Conditions(potencias, x))

def is_min(f, x0, tol = 1e-25):
    """
    Dada una función f y un punto x0, esta función regresa True si x0 
    es un mínimo, False si no podemos garantizar que lo sea
    """
    # Como F_O_C / S_O_C resgresa un bool no necesitamos el == True
    if First_Order_Conditions(f, x0) and Second_Order_Conditions(f, x0, tol):
        return True
    else:
        return False

x = [1,1]
print(is_min(potencias, x)) # FALSE
print(First_Order_Conditions(potencias, x)) # FALSE
print(Second_Order_Conditions(potencias, x)) # TRUE

x = [0,0]
print(is_min(potencias, x)) # TRUE
print(First_Order_Conditions(potencias, x)) # TRUE
print(Second_Order_Conditions(potencias, x)) # TRUE

x = [0.046656, 0.046656]
print(is_min(potencias, x)) 

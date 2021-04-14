import numpy as np 
from Clase_22_Febrero import Grad,Hessiana 
from Clase_25_Enero import cuadrados
from Clase_1_Marzo import condiciones_wolfe

def genera_alpha(f,x0,pk,c1=1e-4,c2=1e-2,tol=1e-5):
    """
    Backtracking LS i.e algoritmo que encuentra una alpha que cumpla con las condiciones de Wolfe.
    """
    alpha, rho, c=1, 4/5, c1
    while f(x0+alpha*pk) > f(x0) + c*alpha*np.dot(Grad(f,x0),pk):
        alpha *= rho 
    return alpha

def is_pos_def(Hessiana):
    #¿Qué significa que la matriz sea lo suficientemente positiva definida?
    return True

def modificacion_hessiana(Hessiana, l =.5):
    while not is_pos_def(Hessiana):
        Hessiana = Hessiana + l*np.eye(len(Hessiana))
    return Hessiana

if __name__ == '__main__':
    alpha = genera_alpha(cuadrados, np.array([1,1,1,1]), np.array([-1,-1,-1,-1]))
    print(condiciones_wolfe(cuadrados, np.array([1,1,1,1]), alpha, np.array([-1,-1,-1,-1])))
    

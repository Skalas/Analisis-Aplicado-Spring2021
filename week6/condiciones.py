from gradiente import gradiente, cuadrados
import numpy as np

def f_o_c(f,x, tol=1e-12):
    """
    Función que calcula las condiciones de primer orden
    """
    grad = np.array(gradiente(f,x))
    if np.dot(grad, grad) < tol:
        return True
    else :
        return False

def s_o_c(f,x,tol=1e-5):
    """
    Inserten aqui código para condiciones de segundo orden 
    """
    
print(f_o_c(cuadrados, [0,0,0,0,0,0,0,0]))

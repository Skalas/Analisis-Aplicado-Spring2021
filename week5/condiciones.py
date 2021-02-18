from gradiente import gradiente, cuadrados
import numpy as np

def f_o_c(f,x, tol=1e-12):
    """
    Función que calcula las condiciones de primer orden
    """
    grad = np.array(gradiente(f,x))
    print(grad)
    if np.dot(grad, grad) < tol:
        return True
    else :
        return False
    
print(condiciones_primer_orden(cuadrados, [0,0,0,0]))

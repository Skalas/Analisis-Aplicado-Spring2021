import numpy as np
from derivadas import Grad, cuadrados, Hess


def f_o_c(f,x, tol=1e-12):
    """
    Función que calcula las condiciones de primer orden
    """
    grad = np.array(Grad(f,x))
    if np.dot(grad, grad) < tol:
        return True
    else :
        return False

def s_o_c(f, x0, tol=1e-15):
    """
    Inserten aqui código para condiciones de segundo orden 
    """
    hess = Hess(f, x0, tol)
    print(np.linalg.eigvals(hess))
    if np.all(np.linalg.eigvals(hess) > tol) :
        return True
    else :
        return False


def is_min(f, x0, tol=1e-25) :
    """
    Dado una función, la función is_min nos va a regresar True si es un mínimo, 
    False si no podemos garantizar que es un mínimo
    """
    if f_o_c(f, x0) and s_o_c(f, x0, tol) :
        return True
    else :
        return False

if __name__ == '__main__':
    print(is_min(cuadrados, [0,1,1,0]))

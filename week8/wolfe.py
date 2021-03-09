import numpy as np 
from derivadas import Grad, Hess, cuadrados

def condiciones_wolfe(f, x0, alpha, pk, c1=1e-4, c2 = 1e-2, tol=1e-5):
    """
    Función que evalúa las condiciones de wolfe para una alpha. 
    f:  función que optimizamos
    x0: punto anterior un numpy.array
    alpha: valor que cumplirá condiciones de wolfe. 
    pk: dirección de decenso un numpy.array
    """
    grad = lambda alpha: Grad(f,x0+alpha*pk, tol)
    phi = lambda alpha: f(x0 + alpha*pk) # Ojo que phi(0) = f(x0)
    linea = lambda alpha: phi(0) + c1 * alpha *np.dot( g_x0, pk)
    g_x0 = grad(0) # grad(0) = Grad(f,x0)
    cond_1 = linea(alpha) - phi(alpha) >= 0
    cond_2 = np.dot(grad(alpha), pk) - c2 * np.dot(g_x0, pk) >=0
    return  cond_1 and cond_2 


if __name__ == '__main__':
    print(condiciones_wolfe(cuadrados, np.array([1,1,1,1]), 1, np.array([-1,-1,-1,-1])))


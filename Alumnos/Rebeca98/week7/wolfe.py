import numpy as np 
from derivadas import cuadrados
from derivadas import Grad,Hess

def condiciones_wolfe(f, x0, alpha, pk, c1=1e-4, c2 = 1e-2,tol=1e-5):
    phi = lambda alpha : f(x0 + alpha*pk) #lambda es una funcion inline(para simplificar la defincion de la funcion y poder evaluar facilmente)
    linea = lambda alpha : f(x0) + c1*alpha*np.dot(Grad(f,x0)*pk)
    cond_1 = linea(alpha) - phi(alpha) >= 0
    cond_2 = np.dot(Grad(f,x0 + alpha*pk)*pk)-c2*np.dot(Grad(f,x0)*pk >= 0
    return linea(alpha) - phi(alpha) >= 0 and 
    


if __name__ == '__main__':
    print(condiciones_wolfe(cuadrados, [1,1,1,1], 1, [-1,-1,-1,-1]))

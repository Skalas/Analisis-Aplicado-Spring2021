import numpy as np 
from derivadas import cuadrados

from derivadas import Grad,Hess

def condiciones_wolfe(f, x0, alpha, pk, tol=1e-5):
    phi = lambda alpha : f(x0 + alpha*pk) #lambda es una función in-line (para simplificar la definición de la función y para poder evaluar más fácilmente)
    linea = lambda alpha : f(x0) + c1*alpha*Grad(x0)*pk
    return linea(alpha) - phi(alpha) >= 0 and Grad(x0+alpha*pk)*pk-c2*Grad(x0)*pk >=0
    #falta transponer 

if __name__ == '__main__':
    print(condiciones_wolfe(cuadrados, [1,1,1,1], 1, [-1,-1,-1,-1]))


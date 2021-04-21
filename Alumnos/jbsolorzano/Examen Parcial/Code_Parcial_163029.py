import numpy as np
from derivadas import Grad, Hess, cuadrados, locochon
from condiciones import f_o_c, s_o_c, is_min
from wolfe import modificacion_hessiana, is_pos_def
from busqueda_lineal import busqueda_lineal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


'Inciso 1'

def rosenbrock(x, a=1, b=100):
    return (a-x[0])**2 + b*(x[0]-x[1]**2)**2

'Inciso 2'
#Algoritmo de Búsqueda Lineal Newton
#Newton ignora las alfas, eventualmente llegan a ser uno
def newton(f, x0):
    xk=x0
    while not (is_min(f,xk)):
        grad = Grad(f,xk)
        hess = modificacion_hessiana(Hess(f,xk))
        pk = np.linalg.solve(hess,-grad)
        xk = xk + pk
    return xk

'Inciso 3'
print(busqueda_lineal(rosenbrock,[4,2]))



#Pruebas códigos
if __name__ == '__main__':
    print('\nOptimizacion de Newton')
    print(newton(rosenbrock,[4,2]))
    print('\nOptimizacion de Busqueda Lineal')
    print(busqueda_lineal(rosenbrock,[4,2]))

import numpy as np
from derivadas import Grad, Hess, cuadrados, locochon
from condiciones import f_o_c, s_o_c, is_min
from wolfe import genera_alpha, modificacion_hessiana, is_pos_def


def busqueda_lineal(f, x0, method ='Maximo Descenso'):
    xk=x0
    while not (is_min(f,xk)):
        grad = Grad(f,xk)
        hess = Hess(f,xk)
        if method == 'Newton':
            hess = modificacion_hessiana(hess)
            pk = np.linalg.solve(hess,-grad)
        elif method == 'Maximo Descenso' :
            pk = -grad
        alpha = genera_alpha(f,xk,pk)
        xk = xk + alpha*pk
    return xk


if __name__ == '__main__':
    #min1 = busqueda_lineal(cuadrados,[1,0,9,0,0], 'Newton')
    #min2 = busqueda_lineal(cuadrados,[1,0,9,0,0])
    min1 = busqueda_lineal(locochon,[6,3],'Newton')
    min2 = busqueda_lineal(locochon,[1,1])
    print('\nCon Newton')
    print(min1)
    print('\nCon Maximo Descenso')
    print(min2)

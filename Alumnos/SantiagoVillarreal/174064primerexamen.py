import numpy as np 
from numpy import linalg as LA
from wolfe import genera_alpha, is_pos_def, modificacion_hessiana
from condiciones import s_o_c, f_o_c
from derivadas import Grad, cuadrados, Hess

"funcion de rosenbrok: f(x,y)=(a-x)^2 + b*(y-x^2)^2"

"a) implementar función"
def Rosen(x0):
    "Funcion de rosenbrok con los parámetros que gustes"
    a=2
    b=5
    Rosen=(a-x0[0])**2 + b*(x0[1]-x0[0]**2)**2
    return Rosen

print(Rosen([1,2]))

"b) metodo de newton "

def BusquedaLineal(f, x0, metodo="Newton"):
    xk=x0
    if metodo == "Newton":
        while not (f_o_c(f,xk)) and (s_o_c(f,xk)):
            grad=Grad(f, xk)
            hess=Hess(f,xk)
            pk=LA.solve(hess,-grad)
            alpha = genera_alpha(f,x0,pk)
            xk= xk + alpha*pk
    else:
        while not (f_o_c(f,xk)) and (s_o_c(f,xk)):
            grad=Grad(f,xk)
            pk = -grad
            alpha = genera_alpha(f,xk,pk)
            xk = xk + alpha*pk
    return xk

print( BusquedaLineal(Rosen,[3,2],"maximo descenso"))

" c) busqueda lineal "
print( BusquedaLineal(Rosen,[3,2],"maximo descenso"))



import numpy as np
from numpy import linalg as LA
from condiciones import s_o_c, f_o_c
from derivadas import Grad, Hess, cuadrados
from wolfe import genera_alpha, is_pos_def, modificacion_hessiana

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

#A continuación está el código de los algoritmos de máximo descenso y Newton, programados anteriormente
##  Algoritmo 3.2: Line search Newton with modification

def NewMod(f,xk,h,maxIt=100):       
    Bk = Hess(f,xk,h)   
    for k in range(maxIt):

        while not is_pos_def(Bk):
            Bk=modificacion_hessiana(Bk)

        pk = LA.inv(Bk).dot(-Grad(f,xk))
        a = genera_alpha(1,f,xk,pk,h)
        xk = xk + a*pk

    return xk


def BusquedaLineal_amplio(f, x0, metodo="maximo descenso"):
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

#Definamos los parámetros a y b, los cuales puede cambiar cuando se desee
a = 5
b = 10

#Definamos la función:
def rosenbrock(x_0):
    res = (a-x_0[0])**2 + b*(x_0[1] - x_[0][0]**2)**2
    return res


#Usando el algoritmo de Newton resulta:
print(BusquedaLineal_amplio(rosenbrock,[1,1], "Newton")

#Usando el algoritmo de búsqueda lineal resulta:
print(BusquedaLineal_amplio(rosenbrock,[1,1], "maximo descenso")

#A continuación está el código para grafiacar la función de Rosenbrock:
     
figRos = plt.figure(figsize=(12, 7))
axRos = figRos.gca(projection='3d')

# Evaluar la función
X = np.arange(-2, 2, 0.15)
Y = np.arange(-1, 3, 0.15)
X, Y = np.meshgrid(X, Y)
Z = f(X,Y)

# Graficar la superficie
surf = axRos.plot_surface(X, Y, Z, cmap=cm.gist_heat_r,
                       linewidth=0, antialiased=False)
axRos.set_zlim(0, 200)
figRos.colorbar(surf, shrink=0.5, aspect=10)
plt.show()
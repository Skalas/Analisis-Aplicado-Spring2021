import numpy as np
from numpy import linalg as LA
from condiciones import s_o_c, f_o_c
from derivadas import Grad, Hess, cuadrados
from wolfe import genera_alpha, is_pos_def, modificacion_hessiana

#Función de costos

b = 100
a = 10
def Rosen (x,y):
 (a-x)*2 + b(y-x*2)*2

#Método de Newton

def NewMet(f,xk,h,It=100):       
    Bk = Hess(f,xk,h)   
    for k in range(It):

        while not is_pos_def(Bk):
            Bk=modificacion_hessiana(Bk)

        pk = LA.inv(Bk).dot(-Grad(f,xk))
        a = genera_alpha(1,f,xk,pk,h)
        xk = xk + a*pk

    return xk



#Método de Búsqueda Lineal

def BusquedaLineal(f, x0, metodo="maximo descenso"):
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

#Extra

figura = plt.figure(figsize=(12, 7))
ax = figura.gca(projection='3d')


X = np.arange(-2, 2, 0.15)
Y = np.arange(-1, 3, 0.15)
X, Y = np.meshgrid(X, Y)
Z = Rosen(X,Y)


surf = axRos.plot_surface(X, Y, Z, cmap=cm.gist_heat_r,
                       linewidth=0, antialiased=False)
ax.set_zlim(0, 200)
figura.colorbar(surf, shrink=0.5, aspect=10)
plt.show()

#La función de Rosenbrock es difícil de optimizar por su forma de valle, es decir, podemos encontrar direcciones de decenso mínimas
#La convergencia al mínimo se "estanca", ya que el mínimo se vuelve difícil de encontrar
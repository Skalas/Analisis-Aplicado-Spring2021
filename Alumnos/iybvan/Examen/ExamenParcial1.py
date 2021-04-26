import numpy as np
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl


"""
CÓDIGOS VISTOS EN CLASE
"""

"""
Condiciones de Primer Orden
"""
def f_o_c(f,x, tol=1e-12):
    grad = np.array(Grad(f,x))
    if np.dot(grad, grad) < tol:
        return True
    else :
        return False

"""
Condiciones de Segundo Orden
"""
def s_o_c(f, x0, tol=1e-15):
    hess = Hess(f, x0, tol)
    print(np.linalg.eigvals(hess))
    if np.all(np.linalg.eigvals(hess) > tol) :
        return True
    else :
        return False


"""
Función que calcula el Grad de una función en un punto
"""
def Grad(f, x0, h=1e-6, i=-1):
    n = len(x0)
    if i in range(n):
        z = np.zeros(n)
        z[i] = h/2
        Grad = (f(x0 + z) - f(x0 - z))/h
    else:
        Grad=np.zeros(n)
        for j in range(n):
            z = np.zeros(n)
            z[j] = h/2
            Grad[j]= (f(x0 + z) - f(x0 - z))/h
    return Grad


"""
Función que calcula la Hessiana  de una función en un punto. 
"""
def Hess(f, x0, h=1e-4, method = "basic"):
    n = len(x0)
    Hess = np.matrix(np.zeros((n,n)))
    for i in range(n):
        for j in range(n):
            z_i = np.zeros(n)
            z_i[i] = h
            z_j = np.zeros(n)
            z_j[j] = h
            if method == "basic":
                Hess[i,j] = ( f(x0 + z_j +z_i) - f(x0 + z_i ) - f(x0+z_j) +f(x0)) / (h**2)
            elif method == "grad":
                Hess[i,j] = (Grad(f,x0+z_j,h,i) - Grad(f,x0,h,i) + \
                             Grad(f,x0+z_i,h,j) - Grad(f,x0,h,j))/(2*h)
            elif method == "centered":
                if i==j:
                    Hess[i,j] = (-f(x0+2*z_i) + 16*f(x0+z_i) - 30*f(x0)+\
                                 16*f(x0-z_i) - f(x0-2*z_i))  / (12*h**2)
                else :
                    Hess[i,j] = (f(x0+z_i+z_j) - f(x0 + z_i - z_j) - \
                                 f(x0 - z_i + z_j) + f(x0-z_i-z_j))/(4*h**2)
            elif method == "gradCentered":
                    Hess[i,j] = (Grad(f,x0+z_j,h)[i] - Grad(f, x0-z_j,h)[i] + \
                                 Grad(f,x0+z_i,h)[j] - Grad(f,x0-z_i,h)[j])/(4*h)
    return Hess


"""
Backtracking LS i.e. Algoritmo que encuentra una alpha que cumpla condiciones de wolfe. 
"""
def genera_alpha(f, x0, pk, c1=1e-4, tol=1e-5):
    alpha, rho, c = 1, 4/5, c1
    while f(x0 + alpha*pk)>f(x0) + c*alpha*np.dot(Grad(f, x0),pk):
        alpha*=rho
    return alpha



"""
DEFINICIÓN DE LA FUNCIÓN
"""

def rosenbrock(x,a=3,b=100):
    z=(a-x[0])**2+b*(x[1]-x[0]**2)**2
    return z



"""
IMPLEMENTACIÓN DE BÚSQUEDA LINEAL
"""

"""
Algoritmo de Newton corte de paso
"""

x0=[0,0]
def BL_Newton_sinalpha(f,x0):
    xk=x0
    while not (f_o_c(f,xk) and s_o_c(f,xk)):
        g=Grad(f,xk)
        h=Hess(f,xk)
        pk=linalg.solve(h,-g)
        xk=xk+pk
    return xk

BL_Newton_sinalpha(rosenbrock,x0)


"""
Algoritmo de Newton
"""

x0=[0,0]
def BL_Newton(f,x0):
    xk=x0
    while not (f_o_c(f,xk) and s_o_c(f,xk)):
        g=Grad(f,xk)
        h=Hess(f,xk)
        pk=linalg.solve(h,-g)
        alpha=genera_alpha(f,x0,pk)
        xk+=alpha*pk
    return xk

BL_Newton(rosenbrock,x0)


"""
Algoritmo de Máximo Descenso
"""

def BL_MD(f,x0):
    xk=x0
    while not (f_o_c(f,xk) and s_o_c(f,xk)):
        g=Grad(f,xk)
        pk=-g
        alpha=genera_alpha(f,x0,pk)
        xk+=alpha*pk
    return xk

BL_MD(rosenbrock,x0)


"""
GRÁFICA DE LA FUNCIÓN
"""

fig = pl.figure()
ax = Axes3D(fig)
X = np.arange(-1.5, 2, 0.1)
Y = np.arange(-0.5, 3, 0.1)
X, Y = np.meshgrid(X, Y)


Z = (1-X)**2+100*(Y-X**2)**2

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=pl.cm.hot)
ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=pl.cm.hot)
ax.set_zlim(0, 1000)

pl.show()
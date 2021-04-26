#

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la

# Definimos

#def Rosenbrock(x0, a = 1, b = 100):
    #r = ((a - x)**2) + (b*((y - x)**2))
    #return r

# Usamos a=1, b=100 por simplicidad para definir la función de Rosenbrock
a = 1
b = 100
f_Rosenbrock = lambda x,y: a*(x-1)**2 + b*(y-x**2)**2


# Usaremos:

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
    hess = Hess(f, x0, tol)
    print(np.linalg.eigvals(hess))
    if np.all(np.linalg.eigvals(hess) > tol) :
        return True
    else :
        return False


def is_min(f, x0, tol=1e-25) :
    if f_o_c(f, x0) and s_o_c(f, x0, tol) :
        return True
    else :
        return False

def condiciones_wolfe(f, x0, alpha, pk, c1=1e-4, c2 = 1e-2, tol=1e-5):
    grad = lambda alpha: Grad(f,x0+alpha*pk, tol)
    phi = lambda alpha: f(x0 + alpha*pk) # Ojo que phi(0) = f(x0)
    linea = lambda alpha: phi(0) + c1 * alpha *np.dot( g_x0, pk)
    g_x0 = grad(0) # grad(0) = Grad(f,x0)
    cond_1 = linea(alpha) - phi(alpha) >= 0
    cond_2 = np.dot(grad(alpha), pk) - c2 * np.dot(g_x0, pk) >=0
    return  cond_1 and cond_2 


def genera_alpha(f, x0, pk, c1=1e-4, tol=1e-5):
    alpha, rho, c = 1, 4/5, c1
    while f(x0 + alpha*pk)>f(x0) + c*alpha*np.dot(Grad(f, x0),pk):
        alpha*=rho
    return alpha



def line_search(f, x0, metodo = "maximo descenso"):
    xk=x0
    if metodo == "Newton":
        while not (f_o_c(f, xk)) and (s_o_c(f, xk)):
            grad=Grad(f, xk)
            hess=Hess(f, xk)
            pk=la.solve(hess, -grad)
            alpha = genera_alpha(f, xk, pk)
            xk= xk + alpha*pk
    else:
        while not (f_o_c(f, xk)) and (s_o_c(f, xk)):
            grad=Grad(f, xk)
            pk = -grad
            alpha = genera_alpha(f, xk, pk)
            xk = xk + alpha*pk
    return xk


# Respuesta a)

#evaluamos en x = (1,2)
print(f_Rosenbrock(1,2)) #coincide


# Respuesta b)

#Damos x0=
x_resp = line_search(f_Rosenbrock , [0,90], metodo = "Newton")


# Respuesta c)

x_resp_c = line_search(f_Rosenbrock, [0,90])


# Extra

#Graficar:

fig = plt.figure(figsize=(12, 7))
ax = fig.gca(projection='3d')

X = np.arange(-2, 2, 0.15)
Y = np.arange(-1, 3, 0.15)
X, Y = np.meshgrid(X, Y)
Z = f_Rosenbrock(X,Y)

surf = ax.plot_surface(X, Y, Z, cmap=cm.gist_heat_r,
                       linewidth=0, antialiased=False)
ax.set_zlim(0, 200)
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()




# codigo del primer parcial 

import numpy as np
from numpy import linalg as LA

#from condiciones import s_o_c, f_o_c
#from derivadas import Grad, Hess, cuadrados
#from wolfe import genera_alpha, is_pos_def, modificacion_hessiana

#agregamos las funciones que nos ayudaran a correr nuestro algoritmo

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
    """
    Inserten aqui código para condiciones de segundo orden 
    """
    hess = Hess(f, x0, tol)
    print(LA.eigvals(hess))
    if np.all(LA.eigvals(hess) > tol) :
        return True
    else :
        return False

def Grad(f, x0, h=1e-6, i=-1):
    """
    Función que calcula el Grad de una función en un punto
    """
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
    """
    Función que calcula la Hessiana  de una función en un punto. 
    f: función sobre la cual queremos calcular la hessiana.
    x0: Punto sobre el cual queremos hacer el cálculo
    h: nivel de precisión para hacer el cálculo
    method: Método por el cual se quiere hacer puede ser: 'basic', 'grad', 'centered', 'gradCentered'
    """
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

def genera_alpha(f, x0, pk, c1=1e-4, tol=1e-5):
    """
    Backtracking LS i.e. Algoritmo que encuentra una alpha que cumpla condiciones de wolfe. 
    """
    alpha, rho, c = 1, 4/5, c1
    while f(x0 + alpha*pk)>f(x0) + c*alpha*np.dot(Grad(f, x0),pk):
        alpha*=rho
    return alpha

#definimos la función de Rosenbrock 

#argumento de la función es un vector
def Rosenbrock(x0):                      
        a=0
        b=100
        x=x0[0]
        y=x0[1]
        f = (a-x)**2 + b*(y-x**2)**2
        return f


# probemos con los siguientes valores para x0
x0=(1.8,2.3)

#Probemos el algoritmo de Newton, es decir el paso alfa es completo=1 y la matriz B es la hessiana 

def BusquedaLineal_alfa1(f, x0, metodo="maximo descenso"):
    xk=x0
    if metodo == "Newton":
        while not (f_o_c(f,xk)) and (s_o_c(f,xk)):
            grad=Grad(f, xk)
            hess=Hess(f,xk)
            pk=LA.solve(hess,-grad)
            alpha = 1
            xk= xk + alpha*pk
    else:
        while not (f_o_c(f,xk)) and (s_o_c(f,xk)):
            grad=Grad(f,xk)
            pk = -grad
            alpha = 1
            xk = xk + alpha*pk
    return xk

print("Este es el resultado obtenido con el algoritmo de Newton:")
print(BusquedaLineal_alfa1(Rosenbrock,x0,0.000001))

#ahora probamos el algoritmo de BLS de Newton en especial

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


print("Este es el resultado obtenido con el algoritmo BLS de Newton:")
print(BusquedaLineal_amplio(Rosenbrock,x0,0.000001))
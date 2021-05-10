
#Primero empezaremos por implementar la función de Rosenbrock
#Recordemos que la función de Rosenbrock se ve de la forma:
#f(x, y) = (a-x)^2 + b(y - x^2)^2

import numpy as np
from numpy import linalg as LA




#Para este ejercicio usaremos a = 1, b = 5
a = 1
b = 5

Rosen = lambda x,y: (a-x)^2 + b(y - x**2)**2
x0 = (2,2)
h = 10**-8


#intentaremos encontrar el mínimo a través del método de Newton
#Pero primero definiré todas las otras funciones que serán utilizados tanto por el método de Newton
#como por el método de Búsqueda Lineal.
#estas funciones ya las habíamos programado en clase.


def is_pos_def(Hessiana):
    """ A Rellenar """
    return True


def modificacion_hessiana(Hessiana, l = 0.5):
    while not is_pos_def(Hessiana):
        Hessiana = Hessiana + l*np.eye(len(Hessiana))
    return Hessiana

def genera_alpha(f, x0, pk, c1=1e-4, tol=1e-5):
    """
    Backtracking LS i.e. Algoritmo que encuentra una alpha que cumpla condiciones de wolfe. 
    """
    alpha, rho, c = 1, 4/5, c1
    while f(x0 + alpha*pk)>f(x0) + c*alpha*np.dot(Grad(f, x0),pk):
        alpha*=rho
    return alpha

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

def cuadrados(x):
    resultado=0
    for i in range(len(x)):
        resultado += x[i]**4
    return resultado

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
    print(np.linalg.eigvals(hess))
    if np.all(np.linalg.eigvals(hess) > tol) :
        return True
    else :
        return False

def NewMod(f,xk,h,maxIt=100):       
    Bk = Hess(f,xk,h)   
    for k in range(maxIt):

        while not is_pos_def(Bk):
            Bk=modificacion_hessiana(Bk)

        pk = LA.inv(Bk).dot(-Grad(f,xk))
        alpha = genera_alpha(1,f,xk,pk,h)
        xk = xk + alpha*pk

    return xk
print(NewMod(Rosen,x0, h))


## Ahora usaremos el método de Búsqueda lineal para resolver el problema de minimiza


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

print(BusquedaLineal_amplio(Rosen, x0))

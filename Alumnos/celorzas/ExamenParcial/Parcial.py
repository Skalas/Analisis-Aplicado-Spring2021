"""
CAROLINA ELORZA SÁNCHEZ       166780
"""

"""
Los siguientes códigos se hicieron a lo largo del semestre y nos van a ayudar para el ejercicio 3
--> Nada más copié y pegué de mis notas
"""
import numpy as np
import math 

def Grad(f,x0,h=1e-6, i=-1): #función gradiente renovada
    n = len(x0)
    if i in range(n):
        z = np.zeros(n)
        z[i] = h/2
        grad = (f(x0+z)-f(x0-z))/h
    else:
        grad = np.zeros(n)
        for j in range(n):
            z = np.zeros(n)
            z[j] = h/2
            grad[j] = (f(x0+z)-f(x0-z))/h
    return grad

def CondPrimerOrden(f,x0,tolerancia=1e-5):
    grad=Grad(f,x0)
    normaCuadrado=np.dot(grad,grad)
    if normaCuadrado<=tolerancia:
        return True
    else:
        return False

def Hessiana(f,x0,h=1e-4, method = "basic"):
    n =  len(x0)
    Hess = np.matrix(np.zeros((n,n)))
    for i in range(n):
        for j in range(n):
            z_i = np.zeros(n)
            z_i[i] = h
            z_j = np.zeros(n)
            z_j[j] = h
            if method == "basic":
                Hess[i,j] = ( f(x0 + z_j + z_i) - f(x0 + z_i) - f(x0 + z_j) + f(x0))/(h**2)
            elif method == "grad":
                Hess[i,j] = (Grad(f,x0+z_j,h,i) - Grad(f,x0,h,i) + Grad(f,x0+z_i,h,j) - Grad(f,x0,h,j))/(2*h)
            elif method == "centered":
                if i==j:
                    Hess[i,j] = (-f(x0+2*z_i) + 16*f(x0+z_i) - 30*f(x0) + 16*f(x0-z_i) - f(x0-2*z_i))/(12*h**2)
                else:
                    Hess[i,j] = (f(x0+z_i+z_j) - f(x0 + z_i - z_j) - f(x0 - z_i + z_j)+ f(x0-z_i-z_j))/(4*h**2)
            elif method == "gradCentered":
                Hess[i,j] = (Grad(f,x0+z_j,h)[i] - Grad(f,x0-z_j,h)[i] +\
                     Grad(f,x0+z_i,h)[j] - Grad(f,x0-z_i,h)[j])/(4*h)
    return Hess

def CondSegOrden(f,x0,tol=1e-5):
    hess = Hessiana(f,x0)
    if np.all(np.linalg.eigvals(hess) > tol) : 
        return True
    else:
        return False

def genera_alpha(f,x0,pk,c1=1e-4,c2=1e-2,tol=1e-5):
    alpha, rho, c=1, 4/5, c1 
    while f(x0+alpha*pk) > f(x0) + c*alpha*np.dot(Grad(f,x0),pk):
        alpha *= rho #la alpha generada cumple con las condiciones de Wolfe
    return alpha

def is_min(f,x0,tol=0.0001):
    """
    Dado una función en un punto, la función is_min nos va a regresar True si es un mínimo, False si
    no podemos garantizar que sea un mínimo 
    """
    if CondPrimerOrden(f,x0) and CondSegOrden(f,x0):
        return True
    else:
        return False

def Busq_Lin(f, x0, quest):
    xk = x0
    while is_min(f,xk) == False:
        if quest == 0: #0 para max descenso y 1 para newton
            grad = Grad(f,xk)
            hess = Hessiana(f,xk)
            pk = np.linalg.solve(hess, -grad)
            alpha = genera_alpha(f,xk,pk)
            xk = xk + alpha*pk
        else: 
            grad = Grad(f,xk)
            pk = -grad
            alpha = genera_alpha(f,xk,pk)
            xk = xk + alpha*pk
    return xk



"""
1. Implementar una función de costo para poderla optimizar
    OJO --> bidimensional, esto es, podemos ver a dos valores (x, y) o un vector con dos entradas, nada más
"""
def Rosenbrock(x): #Approach donde x es vector de dos entradas
    a=1
    b=100
    resultado = (a-x[0])**2 + b*((x[1]-x[0]**2)**2)
    return resultado

"""
2. Usar el algoritmo de Newton para intentar optimizarla 
    --> algoritmo de búsqueda lineal pero sin necesidad de encontrar la alpha
"""
def Alg_Newt(f, x0): #xk vector de dos dimensiones
    xk = x0
    while is_min(f,xk) == False:
        grad = Grad(f,xk)
        hess = Hessiana(f,xk)
        pk = np.linalg.solve(hess, -grad)
        xk = xk + pk 
    #Se vio en clase que el alg. de Newton empieza a ignorar las alpha's, esto es, 
    #eventualmente llegan a ser uno (Clase 1ero de marzo)
    return xk

print(Alg_Newt(Rosenbrock,[.3,2])) 

"""
3. Usar el algoritmo de búsqueda lineal para optimizarla
"""
print(Busq_Lin(Rosenbrock,[.3,2],0)) #Busq_lin bajo Max Descenso
print(Busq_Lin(Rosenbrock,[.3,2],1)) #Busq_lin bajo Newton

"""
Hagan la gráfica de la función para entender por qué es díficil
    --> googlée cómo se hace esto porque no tenía el apunte completo de las gráficas
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

x = [[np.linspace(0,20,100)],[np.linspace(0,20,100)]]
plt.plot(x, Rosenbrock)
plt.show()


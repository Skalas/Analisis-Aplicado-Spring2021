#Importación de librerías
import numpy as np
from numpy import linalg as LA
import random

#FUNCIONES HECHAS EN CLASE
def cuadrados(x):
    resultado = 0
    for i in range(len(x)):
        resultado += x[i]**2
    return resultado

def Hess(f, x0, h=1e-4, method="basic"):
    """
    Función que calcula la Hessiana  de una función en un punto.
    f: función sobre la cual queremos calcular la hessiana.
    x0: Punto sobre el cual queremos hacer el cálculo
    h: nivel de precisión para hacer el cálculo
    method: Método por el cual se quiere hacer puede ser:
             'basic', 'grad', 'centered', 'gradCentered'
    """
    n = len(x0)
    Hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            z_i = np.zeros(n)
            z_i[i] = h
            z_j = np.zeros(n)
            z_j[j] = h
            if method == "basic":
                Hess[i, j] = (f(x0 + z_j + z_i) - f(x0 + z_i) -
                              f(x0+z_j) + f(x0)) / (h**2)
            elif method == "grad":
                Hess[i, j] = (Grad(f, x0+z_j, h, i) - Grad(f, x0, h, i) +
                              Grad(f, x0+z_i, h, j) - Grad(f, x0, h, j))/(2*h)
            elif method == "centered":
                if i == j:
                    Hess[i, j] = (-f(x0+2*z_i) + 16*f(x0+z_i) - 30*f(x0) +
                                  16*f(x0-z_i) - f(x0-2*z_i)) / (12*h**2)
                else:
                    Hess[i, j] = (f(x0+z_i+z_j) - f(x0 + z_i - z_j) -
                                  f(x0 - z_i + z_j) + f(x0-z_i-z_j))/(4*h**2)
            elif method == "gradCentered":
                Hess[i, j] = (Grad(f, x0+z_j, h)[i] - Grad(f, x0-z_j, h)[i] +
                              Grad(f, x0+z_i, h)[j] - Grad(f, x0-z_i, h)[j])\
                               / (4 * h)
    return Hess

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
        Grad = np.zeros(n)
        for j in range(n):
            z = np.zeros(n)
            z[j] = h/2
            Grad[j] = (f(x0 + z) - f(x0 - z))/h
    return np.array(Grad)

def gradiente_conjugado(x0, A, b):
    xk = x0
    b = np.matrix(b).T
    rk = np.dot(A, x0) - b
    pk = -rk
    while not (rk.T * rk ==  0):
        alphak = rk.T * rk / (pk.T * A * pk)
        alphak= alphak[0,0]
        xk_1 = xk + alphak * pk
        rk_1 =  rk + alphak * A * pk
        betak_1 = (rk_1.T * rk_1) / (rk.T * rk)
        betak_1 = betak_1[0,0]
        pk_1 = -rk_1 + betak_1 * pk
        xk, rk, pk = xk_1, rk_1, pk_1
    return xk

def genera_alpha(f, x0, pk, c1=1e-4, c2 = 0.5, tol=1e-5):
    """
    Backtracking LS i.e. Algoritmo que encuentra una
    alpha que cumpla condiciones de wolfe.
    """
    alpha, rho = 1, 3/4
    Gkpk = Grad(f, x0).dot(pk)
    while f(x0 + alpha*pk) > f(x0) + c1*alpha*Gkpk:
        alpha *= rho
    return alpha

#PREGUNTAS DE EXAMEN
#2.1 
def DFP_Hk(yk, sk, Hk):
    """
    Función que calcula La actualización DFP de la matriz Hk
    In:
      yk: Vector n
      sk: Vector n
      Hk: Matriz nxn
    Out:
      Hk+1: Matriz nxn
    """
    Hk1 = Hk - (np.dot(Hk,np.dot(yk,np.dot(yk.T,Hk))))/(np.dot(yk.T,np.dot(Hk,yk))) + (np.dot(sk,sk.T))/(np.dot(yk.T,sk))
    return Hk1

#La función estaba mal implementada ya que no tenía los productos punto correspondientes

x0 = [(-1)**i*10 for i in range(10) ]

def BFGS_DFP(f, x0, tol, H0, maxiter=10000):
    k = 0
    Gk = Grad(f, x0)
    Hk = H0
    xk = np.array(x0)
    xk1 = np.array(x0)
    sk = np.array(100)
    while (LA.norm(Gk) > tol and LA.norm(sk) > tol and k <= maxiter):
        pk = - Hk.dot(Gk)
        alphak = genera_alpha(f, xk, pk)
        xk1 = xk + alphak * pk
        sk = xk1 - xk
        Gk1 = Grad(f, xk1)
        yk = Gk1 - Gk
        Hk = DFP_Hk(yk, sk, Hk) #<-Modificación al algoritmo de clase
        k += 1
        xk = xk1
        Gk = Gk1
    return xk1, k

B0=Hess(cuadrados,x0)
H0=LA.inv(B0)

x,k=BFGS_DFP(cuadrados,x0,1e-15,H0)
print(f'Óptimo {x}, Iteraciones {k}' )

  Óptimo [-2.13734496e-04  5.45731110e-05 -2.13734496e-04  5.45731110e-05
 -2.13734496e-04  5.45731110e-05 -4.32045099e-05  2.25102774e-04
  1.36397880e-05  2.81946102e-04], Iteraciones 95

#2.2
random.seed(165860) 
Diag_A = [random.randint(1,1000) for x in range(1000000)]
b = [random.randint(1,1000) for x in range(1000000)]


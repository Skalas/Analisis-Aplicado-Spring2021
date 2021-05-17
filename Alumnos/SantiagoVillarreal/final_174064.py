#DFP
import numpy as np
from numpy import linalg as LA

def Grad(f, x0, h=1e-6, i=-1):
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

def Hess(f, x0, h=1e-4, method="basic"):
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

def cuadrados(x):
    resultado = 0
    for i in range(len(x)):
        resultado += x[i]**2
    return resultado

def DFP_Hk(yk, sk, Hk):
    yk = np.array([yk]).T
    sk = np.array([sk]).T


    Hk1 = Hk - (Hk * yk.dot(yk.T) * Hk)/(yk.T * Hk * yk) + (sk.dot(sk.T))/(yk.T.dot(sk))
    return Hk1

def genera_alpha(f, x0, pk, c1=1e-4, c2 = 0.5, tol=1e-5):
    alpha, rho = 1, 3/4
    Gkpk = Grad(f, x0).dot(pk)
    while f(x0 + alpha*pk) > f(x0) + c1*alpha*Gkpk:
        alpha *= rho
    return alpha

def BFGS(f, x0, tol, H0, maxiter=10000):
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
        Hk = DFP_Hk(yk, sk, Hk)
        k += 1
        xk = xk1
        Gk = Gk1
    return xk1, k

x0 = [(-1)**i*10 for i in range(10)]
B0 = Hess(cuadrados, x0)
H0 = LA.inv(B0)
x, k = BFGS(cuadrados, x0, 1e-15, H0)

# Gradiente Conjugado
import numpy as np
import random

cols=1000000
random.seed(174064)
Diag_A=[random.randint(1,1000) for x in range(cols)]
b =[random.randint(1,1000) for x in range(cols)]

#Matriz rala 
filas, columnas = (3, cols)

matriz_resumida = [[0 for i in range(columnas)] for j in range(filas)]

for i in range(10):
    matriz_resumida[0][i]=i
    matriz_resumida[1][i]=i
    matriz_resumida[2][i]=Diag_A[i]

#No logré que corriera el código completo, intenté alterarlo para que no fuera
#multiplicación de matrices, sino solo el vector diagonal por los otros
#creo que falla por la cuestión de arrgelos y el tipo de variables en matriz
def gradiente_conjugado(x0, A, b):
    xk = x0
    b = np.matrix(b).T
    c=[0 for i in range(len(A[2]))]
    for i in range(len(A[2])):
        c[i]=x0[i]*A[2][i]
    c=np.matrix(c)
    rk = c.T - b
    pk = -rk
    d=[0 for i in range(len(A[2]))]
    while not (rk.T * rk ==  0):
        for i in range(len(A[2])):
            d[i]=pk[i]*A[2][i]
        div=np.dot(pk,d)
        alphak = rk.T * rk / div
        alphak= alphak[0,0]
        xk_1 = xk + alphak * pk
        n=[A[2][i]*pk[i]*alphak for i in range(len(A[2]))]
        rk_1 =  rk + n 
        #Aquí rompe mi código, según yo es por error de tipos de variables o algo similar, creo que arreglando eso debería funcionar, no tuve tiempo para revisarlo :(
        betak_1 = (rk_1.T * rk_1) / (rk.T * rk)
        betak_1 = betak_1[0,0]
        pk_1 = -rk_1 + betak_1 * pk
        xk, rk, pk = xk_1, rk_1, pk_1
    return xk

print(gradiente_conjugado([random.randint(1,1000) for x in range(cols)],matriz_resumida,b))

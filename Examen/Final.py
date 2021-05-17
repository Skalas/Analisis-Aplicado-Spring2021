#Traje todos los códigos a este archivo

import numpy as np
from numpy import linalg as LA

'''
Códigos para las preguntas
'''

'''
Derivadas
'''

import numpy as np


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

'''
Wolfe
'''

def condiciones_wolfe(f, x0, alpha, pk, c1=1e-4, c2 = 1e-2, tol=1e-5):
    """
    Función que evalúa las condiciones de wolfe para una alpha. 
    f:  función que optimizamos
    x0: punto anterior un numpy.array
    alpha: valor que cumplirá condiciones de wolfe. 
    pk: dirección de decenso un numpy.array
    """
    grad = lambda alpha: Grad(f,x0+alpha*pk, tol)
    phi = lambda alpha: f(x0 + alpha*pk) # Ojo que phi(0) = f(x0)
    linea = lambda alpha: phi(0) + c1 * alpha *np.dot( g_x0, pk)
    g_x0 = grad(0) # grad(0) = Grad(f,x0)
    cond_1 = linea(alpha) - phi(alpha) >= 0
    cond_2 = np.dot(grad(alpha), pk) - c2 * np.dot(g_x0, pk) >=0
    return  cond_1 and cond_2 


def genera_alpha(f, x0, pk, c1=1e-4, tol=1e-5):
    """
    Backtracking LS i.e. Algoritmo que encuentra una alpha que cumpla condiciones de wolfe. 
    """
    alpha, rho, c = 1, 4/5, c1
    while f(x0 + alpha*pk)>f(x0) + c*alpha*np.dot(Grad(f, x0),pk):
        alpha*=rho
    return alpha


def is_pos_def(Hessiana):
    """ A Rellenar """
    return True


def modificacion_hessiana(Hessiana, l = 0.5):
    while not is_pos_def(Hessiana):
        Hessiana = Hessiana + l*np.eye(len(Hessiana))
    return Hessiana



'''
Actualizar BFGS
'''

def BFGS_Hk(yk, sk, Hk):
    """
    Función que calcula La actualización BFGS de la matriz Hk
    In:
      yk: Vector n
      sk: Vector n
      Hk: Matriz nxn
    Out:
      Hk+1: Matriz nxn
    """
    n = len(yk)
    yk = np.array([yk]).T
    sk = np.array([sk]).T
    rhok = 1 / yk.T.dot(sk)
    Vk = (np.eye(n) - rhok * yk.dot(sk.T))
    Hk1 = Vk.T * Hk * Vk + rhok * sk.dot(sk.T)
    return Hk1


def BFGS_Bk(yk, sk, Bk):
    """
    Función que calcula La actualización BFGS de la matriz Bk
    In:
      yk: Vector n
      sk: Vector n
      Bk: Matriz nxn
    Out:
      Bk+1: Matriz nxn
    """
    return Bk - (np.dot(Bk, np.dot(sk, np.dot(sk, Bk)))) / (np.dot(sk, np.dot(Bk, sk))) + np.dot(yk, yk) / np.dot(yk, sk)

'''
Código final BFGS
'''


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
        Hk = BFGS_Hk(yk, sk, Hk)
        k += 1
        xk = xk1
        Gk = Gk1
    return xk1, k



'''
PREGUNTA 1

'''

'''
Actualizar DFP
'''

def DFP_Bk(yk, sk, Bk):
    """
    Función que calcula La actualización DFP de la matriz Bk
    In:
      yk: Vector n
      sk: Vector n
      Bk: Matriz nxn
    Out:
      Bk+1: Matriz nxn
    """
    n = len(yk)
    yk = np.array([yk]).T
    sk = np.array([sk]).T
    rhok = 1 / (yk.T.dot(sk))
    Vk = (np.eye(n) - rhok * yk.T * sk)
    Bk1 = Vk.T * Bk * Vk + rhok * yk * yk.T
    return Bk1


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
    Hk1 = Hk - (np.dot(Hk, np.dot(yk, np.dot(yk, Hk))))/(np.dot(yk, np.dot(Hk, yk)))  + np.dot(sk, sk) / np.dot(yk, sk)
    return Hk1

'''
Código final DFP
'''

def DFP(f, x0, tol, B0, maxiter=10000):
    k = 0
    Gk = Grad(f, x0)
    Bk = B0
    xk = np.array(x0)
    xk1 = np.array(x0)
    sk = np.array(100)
    while (LA.norm(Gk) > tol and LA.norm(sk) > tol and k <= maxiter):
        pk = - Bk.dot(Gk)
        alphak = genera_alpha(f, xk, pk)
        xk1 = xk + alphak * pk
        sk = xk1 - xk
        Gk1 = Grad(f, xk1)
        yk = Gk1 - Gk
        Bk = DFP_Bk(yk, sk, Bk)
        k += 1
        xk = xk1
        Gk = Gk1
    return xk1, k




def cuadrados(x):
    resultado = 0
    for i in range(len(x)):
        resultado += x[i]**2
    return resultado


def rosenbrock(x, a=1, b=100):
    return (a-x[0])**2 + b * (x[1] - x[1]**2)**2


x0 = [(-1)**i*10 for i in range(10)]

if __name__ == "__main__":
    #x, k = DFP(rosenbrock, np.array([-100000, 1]), 1e-15, np.eye(2))
    #print(f'Llegué a {x} en {k} iteraciones')
    x, k = DFP(cuadrados, np.array(x0), 1e-15, np.eye(10))
    print(f'Llegué a {x} en {k} iteraciones')    




'''
PREGUNTA 2
'''
import random

'''
Códigos para la pregunta
'''

def generar_conjunto_canonico(n):
    A = np.matrix(np.eye(n)).T
    return [A[i].T for i in range(n)]


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

def gradiente_conjugado_precond(x0, A, b, M):
    xk = x0
    b = np.matrix(b).T
    rk = np.dot(A, x0) - b
    yk = np.linalg.solve(M, rk)
    pk = -yk 
    while not (rk.T * rk ==  0):
        alphak = rk.T * yk / (pk.T * A * pk)
        alphak= alphak[0,0]
        xk_1 = xk + alphak * pk
        rk_1 =  rk + alphak * A * pk
        yk_1 = np.linalg.solve(M, rk_1)
        betak_1 = (rk_1.T * rk_1) / (rk.T * rk)
        betak_1 = betak_1[0,0]
        pk_1 = -yk_1 + betak_1 * pk
        xk, rk, pk, yk  = xk_1, rk_1, pk_1, yk_1
    return xk



#El problema

def matrizRala(Diag_A,i,j):
    A = [Diag_A[i],0,0,Diag_A[j]]
    A = np.array(A).reshape(2,2)
    return A


#Pruebas

if __name__ == '__main__':
    random.seed(173993)
    Diag_A = [random.randint(1,1000) for x in range(1000000)]
    b = [random.randint(1,1000) for x in range(1000000)]
    x0 = np.matrix(np.zeros(1000000)).T
    i = 0

    while (i < 1000000):
        bi = b[i]
        bi1 = b[i+1]
        baux = [bi, bi1]

        aux = matrizRala(Diag_A,i,i+1)
        print(gradiente_conjugado([0,0],aux,baux))
        i = i+2
        






import numpy as np
from numpy import linalg as LA

# Obs: la implementación está abajo de las funciones
# usamos las funciones de clase:

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
    rhok = 1 / (yk.T*sk)
    Vk = (np.eye(n) - rhok * yk*sk.T)
    Bk1 = Vk * Bk * Vk.T + rhok * yk * yk.T
    return Bk1


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


def cuadrados(x):
    resultado = 0
    for i in range(len(x)):
        resultado += x[i]**2
    return resultado


# 2.1

# Corremos el algoritmo con el punto inicial x0 dado
x0 = [(-1)**i*10 for i in range(10)]  
print(x0)

# f va a ser cuadrados (de dimension 10)

def optimo_BFGS(x0, f, tolerancia = 0.001):
    xk = x0
    sk =x0
    yk = Grad(f,x0)
    Hk = Hess(f, x0)
    while LA.norm(Grad(f, xk)) > tolerancia :
        #direccion
        pk = -Hk*Grad(f,xk)
        # Wolfe
        alpha = genera_alpha(f, xk, pk)
        #
        xk_viejo = xk
        xk = xk + alpha*pk
        sk = xk - xk_viejo
        yk = Grad(f, xk) - Grad(f, xk_viejo)
        Hk = BFGS_Hk(yk, sk, Hk)
    
    return xk

#Respuesta:
print("El optimo es: " + optimo_BFGS(x0, cuadrados))


# 2.2

import random
random.seed(174052)
Diag_A = [random.randint(1, 1000) for x in range(1000000)]
b = [random.randint(1, 1000) for x in range(1000000)]

#punto inicial aleatorio
x0_GC = [random.randint(1, 1000) for x in range(1000000)]


# Definamos la matriz rala
# solo vamos a gurdar 3 datos: A(i,j), i, j
# Hacemos un arreglo que contenga esta informacion

arr = np.zeros((len(Diag_A),3))
for i in range(len(arr)):
    #arreglo con el valor de la entrada A(i,j) y los valores de i, j
    arr[i][0] = Diag_A[i]
    # guardamos la posicion i,j
    arr[i][1] = i + 1
    arr[i][2] = i + 1

# Ahora, llamamos a las funciones que vamos a resolver

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


# Manos a la obra

valores_arr = arr[:,0]

# en lugar de hacer las multiplicaciones de la matriz A, vamos a usar el arreglo y hacer producto punto

def implementacion_GC(x0, arr, b):

    #x0, b y arr son los definidos anteriormente

    r0 = np.zeros(len(arr[:,0])) 
    for i in range(len(r0)):
        r0[i] = arr[i][0] - b[i] 
    pk = -r0
    rk = r0
    xk = x0
    while not(LA.norm(rk) ==  0):
        p_cuadrado = pk**2
        alphak = rk.T * rk / (p_cuadrado.T * arr[:,0])
        xk_1 = xk + alphak * pk
        rk_1 =  rk + alphak * (arr[:,0].T * pk)
        betak_1 = (rk_1.T * rk_1) / (rk.T * rk)
        pk_1 = -rk_1 + betak_1 * pk
        xk, rk, pk = xk_1, rk_1, pk_1
    return xk


# Respuesta:
print("La solución es: " + implementacion_GC(x0_GC, arr, b))
    

    
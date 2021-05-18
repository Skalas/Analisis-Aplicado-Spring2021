import numpy as np
import random
# Pregunta 2.1 Corre el Algoritmo BFGS, pero con la actualización DFP con la
# función cuadrados en dimensión 10 con punto inicial:


def cuadrados(x):
    resultado = 0
    for i in range(len(x)):
        resultado += x[i]**4
    return resultado


def DFP_Bk(yk, sk, Bk):
    n = len(yk)
    rhok = 1 / (yk.T*sk)
    Vk = (np.eye(n) - rhok * yk*sk.T)
    Bk1 = Vk * Bk * Vk.T + rhok * yk * yk.T
    return Bk1


def DFP_Hk(yk, sk, Hk):
    Hk1 = Hk - (Hk * yk * yk.T * Hk)/(yk.T * Hk * yk) + (sk * sk.T)/(yk.T * sk)
    return Hk1


def BFGS_Hk(yk, sk, Hk):
    n = len(yk)
    yk = np.array([yk]).T
    sk = np.array([sk]).T
    rhok = 1 / yk.T.dot(sk)
    Vk = (np.eye(n) - rhok * yk.dot(sk.T))
    Hk1 = Vk.T * Hk * Vk + rhok * sk.dot(sk.T)
    return Hk1


def BFGS_Bk(yk, sk, Bk):
    return Bk - (np.dot(Bk, np.dot(sk, np.dot(sk, Bk)))) /
    (np.dot(sk, np.dot(Bk, sk))) + np.dot(yk, yk) / np.dot(yk, sk)


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


x0 = [(-1)**i*10 for i in range(10)]
x, k = BFGS(cuadrados, x0, 1e-15, np.eye(2))


# -------------------------------------------------------------------------

# Pregunta 2.2: Con TU clave única como semilla, genera una matriz diagonal con
# 106 números aleatorios en la diagonal, después de esto, genera un vector del
# mismo tamaño.

random.seed(151767)  # Cambien a su propia clave
Diag_A = [random.randint(1, 1000) for x in range(1000000)]


# SPARSE: Creamos la matriz rala con los datos
a = np.array(range(1000000))
A = np.column_stack((a, a, a, Diag_A))
# donde la primera columan de A es el indice, la segunda y tercera son las
# coordenadas del valor en la matriz original y la cuarta es el valor de la
# diagonal de la matriz original.


def gradiente_conjugado(x0, A, b):
    xk = x0
    b = [random.randint(1, 1000) for x in range(1000000)].T
    rk = np.dot(A, x0) - b
    pk = -rk
    while not (rk.T * rk == 0):
        alphak = rk.T * rk / (pk.T * A * pk)
        alphak = alphak[0, 0]
        xk_1 = xk + alphak * pk
        rk_1 = rk + alphak * A * pk
        betak_1 = (rk_1.T * rk_1) / (rk.T * rk)
        betak_1 = betak_1[0, 0]
        pk_1 = -rk_1 + betak_1 * pk
        xk, rk, pk = xk_1, rk_1, pk_1
    return xk

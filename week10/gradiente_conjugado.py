import numpy as np


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


if __name__ == '__main__':
    n=15
    A = np.matrix(np.eye(n)).T
    b = np.zeros(n)+1
    x0 = np.matrix(np.zeros(n)).T
    print(gradiente_conjugado_precond(x0, A, b, A))


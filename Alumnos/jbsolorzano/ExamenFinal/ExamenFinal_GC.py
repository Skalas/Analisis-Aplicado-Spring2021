import numpy as np
import random

def multiplica(A,x):
    [n,m]= np.asarray(A.shape)
    Ax = np.asarray(np.zeros(n))
    for i in range(n):
        j = 1
        while j < m and abs(A[i,j+1]) < 1e-15:
            Ax[i] = Ax[i] + A[i,j] * x[ int(A[i,j+1])]
            j += 2
    return Ax

def gradiente_conjugado(x0, A, b):
    xk = x0
    b = np.matrix(b).T
    rk = multiplica(A, x0) - np.asarray(b)[0]
    pk = -rk
    maxiter = 1
    while not (np.dot(rk,rk) ==  0  or maxiter >10000):
        alphak = rk.T * rk / (np.dot(pk, multiplica(A , pk)))
        xk_1 = xk + alphak * pk
        rk_1 =  rk + alphak * multiplica(A, pk)
        betak_1 = (rk_1.T * rk_1) / (rk.T * rk)
        pk_1 = -rk_1 + betak_1 * pk
        xk, rk, pk = xk_1, rk_1, pk_1
        maxiter += 1
    return xk



if __name__ == '__main__':
    random.seed(163029)
    Diag_A = [random.randint(1,1000) for x in range(1000000)]
    Diag_A.append([random.randint(1,1000) for x in range(1000000)])
    b = [random.randint(1,1000) for x in range(1000000)]
    x0 = [random.randint(1,1000) for x in range(1000000)]
    print(gradiente_conjugado(x0, Diag_A, b))

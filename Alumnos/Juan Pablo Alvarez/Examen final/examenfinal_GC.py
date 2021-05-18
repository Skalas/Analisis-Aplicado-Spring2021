import random
import numpy as np

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

random.seed(165815) #  Cambien a su propia clave
Diag_A = [random.randint(1,1000) for x in range(1000000)]
b = [random.randint(1,1000) for x in range(1000000)]

row = np.matrix(np.array([i for i in range(1000000)]))
col = np.matrix(np.array([i for i in range(1000000)]))

D =np.matrix(Diag_A)
A = np.vstack((row,col,D))

if __name__ == '__main__':
    x0 = np.matrix(np.zeros(1000000)).T
    print(gradiente_conjugado(x0, A.T, b,))
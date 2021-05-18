import numpy as np
import random

random.seed(171786)
Diag_A = [random.randint(1,1000) for x in range(1000000)]
b = [random.randint(1,1000) for x in range(1000000)]

def gradiente_conjugado(x0, Diag_A, b):
    xk = x0
    jk = np.zeros(1000000)
    for i in range(1000000):
        jk[i] = Diag_A[i] * x0[i]
    rk = jk - b
    pk = -rk
    sk = 0
    for i in range(1000000):
        sk += rk[i]**2
    lk = np.zeros(1000000)
    for i in range (1000000):
        lk[i] = pk[i] * Diag_A[i]
    while not (sk == 0) or (lk.T * pk == 0):
        sk = 0
        for i in range(1000000):
            sk += rk[i]**2
        lk = np.zeros(1000000)
        for i in range (1000000):
            lk[i] = pk[i] * Diag_A[i]
        alphak = sk / (np.dot(lk, pk))
        tk = np.zeros(1000000)
        for i in range(1000000):
            tk[i] = alphak * pk[i]
        xk_1 = xk + tk
        mk = np.zeros(1000000)
        for i in range (1000000):
            mk[i] = Diag_A[i] * pk[i]
        rk_1 =  rk + alphak * mk
        betak_1 = np.dot(rk_1, rk_1) / (sk)
        pk_1 = -rk_1 + betak_1 * pk
        xk = xk_1
        rk = rk_1
        pk = pk_1
    return xk

"""
jk = np.zeros(1000000)
x0 = np.zeros(1000000)
for i in range(1000000):
        jk[i] = Diag_A[i] * x0[i]
rk = jk - b
print(-rk)
sk = 0
for i in range(1000000):
    sk += rk[i]**2
print(sk)
lk = np.zeros(1000000)
pk = -rk
for i in range (1000000):
    lk[i] = pk[i] * Diag_A[i]
alphak = sk / (lk.T * pk)
print(alphak[0])
tk = np.zeros(1000000)
for i in range(1000000):
    tk[i] = alphak[0] * pk[i]
xk_1 = x0 + tk
print(xk_1)
mk = np.zeros(1000000)
for i in range (1000000):
    mk[i] = Diag_A[i] * pk[i]
rk_1 =  rk + alphak[0] * mk
print(rk_1)
betak_1 = (rk_1.T * rk_1) / (sk)
print(betak_1[0])
pk_1 = -rk_1 + betak_1[0] * pk
print(pk_1)
   
"""


if __name__ == "__main__":
    x0 = np.zeros(1000000)
    print(gradiente_conjugado(x0, Diag_A, b))
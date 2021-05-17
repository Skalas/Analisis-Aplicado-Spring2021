import random
import numpy as np


random.seed(131008)
Diag_A=[random.randint(1,1000) for x in range(1000000)]
b=[random.randint(1,1000) for x in range(1000000)]
x0=np.ones(1000000)

def gradiente_conjugado(x0, A, b):
    xk = np.array(x0)
    A = np.array(A)
    b = np.array(b)
    rk=A*xk-b
    pk=-rk
    
    while not (sum(rk * rk) ==  0):
        alphak = -(sum(rk*pk)/sum(A*pk**2))
        xk_1 = xk+alphak*pk
        rk_1 =  A*xk_1-b
        betak_1 = sum(rk_1*A *pk) / sum(A*pk**2)
        pk_1 = -rk_1+betak_1*pk
        xk, rk, pk = xk_1, rk_1, pk_1
    return xk

x=gradiente_conjugado(x0,Diag_A,b)


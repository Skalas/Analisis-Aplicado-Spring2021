import numpy as np


def generar_conjunto_canonico(n):
    A = np.matrix(np.eye(n)).T
    return [A[i].T for i in range(n)]


def gradiente_conjugado(x0, A, b):
    xk = x0
    b = np.matrix(b).T
    rk = np.dot(A,x0) - b
    pk = -rk
    while not (np.dot(rk.T,rk) ==  0):
        alpha_k = np.dot(-rk.T,pk)/np.dot(pk.T,np.dot(A,pk))
        xk_1 = xk + alpha_k[0,0]*pk
        
        rk=np.zeros(1)
    return rk

if __name__ == '__main__':
    n=15
    A = np.matrix(np.eye(n)).T
    b = np.zeros(n)+1
    x0 = np.matrix(np.zeros(n)).T
    print(gradiente_conjugado(x0,A,b))


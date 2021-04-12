# 1:10:51
import numpy as np 
n=15

A = np.matrix(np.eye(n)).T
b = np.zeros(n) + 1

def generar_conjunto_canonico(n):
    A = np.matrix(np.eye(n)).T
    return [A[i].T for i in range(n)]

def direcciones_conjugadas(x0, cc, A, b): #cc son las direcciones
#Todo Fix error on the alpha = 0 
    xk = x0
    for k in range(len(x0)):
        rk = np.matrix(A+xk-b)
        alpha = np.dot(-rk.T,cc[k])/(np.dot(cc[k].T, A*cc[k]))
        xk_1 = xk + alpha[0,0]*cc[k]
        print(xk_1)
    return xk_1

if __name__ == '__master__':
    n=15
    A = np.matrix(np.eye(n)).T
    b = np.zeros(n) + 1
    cc = generar_conjunto_canonico(n)
    x0 = np.matrix(np.zeros(n)).T
    print(direcciones_conjugadas(x0, cc, A, b))

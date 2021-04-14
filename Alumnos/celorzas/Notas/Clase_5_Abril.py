import numpy as np 

def generar_conjunto_canonico(n):
    A = np.matrix(np.eye(n)).T
    return [A[i].T for i in range(n)]

def proto_gradiente_conjugado(x0, A, b): #Versión hecha el 5 de abril 
    xk = x0
    b = np.matrix(b).T
    rk = np.dot(A,x0) - b
    pk = -rk
    while not (np.dot(rk.T,rk) == 0):
        alpha_k = np.dot(-rk.T,pk)/np.dot(pk.T, np.dot(A,pk))
        print(alpha_k)
        xk_1 = xk + alpha_k*pk
        rk=0
    return rk, xk_1   

def gradiente_conjugado(x0, A, b): #Versión mejorada del proto anterior, hecho el 7 de abril
    xk = 0
    b = np.matrix(b).T
    rk = np.dot(A, x0) - b
    pk = -rk
    while not(np.dot(rk.T,rk) == 0):
        alpha_k = rk.T*rk / pk.T*A*pk 
        xk_1 = xk + alpha_k * pk     #xk_1 es igual a x_(k+1)
        rk_1 = rk + alpha_k * A * pk
        betak_1 = rk_1.T * rk_1 /(rk.T*rk)
        betak_1 = betak_1[0, 0]
        pk_1 = -rk_1 + betak_1 + pk 
        xk, rk, pk = xk_1, rk_1, pk_1
    return rk
    #Tarea hacerlo con una estructura de mmjjdks, solo hay que redifinir las multiplicaciones con la matriz A

if __name__ == '__main__': 
    n=15
    A = np.matrix(np.eye(n)).T
    b = np.zeros(n) + 1
    x0 = np.matrix(np.zeros(n)).T
    print(gradiente_conjugado(x0, A, b))
    print(A, b , x0)
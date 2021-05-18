import numpy as np 
from numpy import linalg as LA

#Problema 2.1
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

def cuadrados(x):
resultado = 0
for i in range(len(x)):
    resultado += x[i]**2
return resultado

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

def DFP_Hk(yk, sk, Hk):
"""
Función que calcula La actualización DFP de la matriz Hk
In:
    yk: Vector n
    sk: Vector n
    Hk: Matriz nxn
Out:
    Hk+1: Matriz nxn
    Realizar yk y sk como arrays
"""
yk = np.array([yk]).T
sk = np.array([sk]).T

Hk1 = Hk - (Hk * yk.dot(yk.T) * Hk)/(yk.T * Hk * yk) + (sk.dot(sk.T))/(yk.T.dot(sk))
#De esta forma, no habrá problemas al obtener HK1
return Hk1

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
    #Cambiamos a DFP_Hk.
    Hk1 = DFP_Hk(yk, sk, Hk)
    k += 1
    Hk = Hk1
    xk = xk1
    Gk = Gk1
return xk1, k

if __name__ == "__main__":
x0 = [(-1)**i*10 for i in range(10)]
#Implementaré la inversa de la Hessiana en vez de la identidad como H0.
B0 = Hess(cuadrados, x0)
H0 = LA.inv(B0)
x, k = BFGS(cuadrados, x0 , 1e-15, H0)
print(f'Llegué a {x} en {k} iteraciones')




#Problema 2.2

import random
random.seed(175551)
Diag_A = [random.randint(1,1000) for x in range(1000000)]
b = [random.randint(1,1000) for x in range(1000000)]

#Generaré una matriz de 3*1000000 que guarde filas, columnas, y elementos de la diagonal, respectivamente, 
#En cada una de sus columnas.
numFilas = 3
numColumnas = 1000000
matrizRala = [[0 for i in range(numColumnas)] for j in range(numFilas)]

for i in range(numColumnas):
    matrizRala[0][i] = i
    matrizRala[1][i] = i
    matrizRala[2][i] = Diag_A[i]
#Hemos guardado los datos importantes de la matriz, sin almacenar todos los ceros.

#Implementemos gradiente conjugado, tendremos que modificarlo
def gradiente_conjugado(x0, b, A = matrizRala):
    xk = x0
    b = np.matrix(b).T
    #Rediseñemos rk:
    #Rediseñemos el producto Ax0
    vec = [0 for i in range(len(matrizRala[0]))]
    rk = [0 for i in range(len(matrizRala[0]))]
    for i in range(len(matrizRala[0])):
        vec[i] = xk[i]*matrizRala[2][i]
        rk[i] = vec[i] - b[i]
    pk = [-e1 for e1, in zip(rk)]
    rk = np.array([rk]).T
    pk = np.array([pk]).T
    while not (rk.T * rk ==  0):
        alphak = rk.T * rk / (pk.T * matrizRala[2] * pk) #Pues matrizRala[2] tiene los elementos de la diagonal.
        alphak= alphak[0,0]
        xk_1 = xk + alphak * pk
        rk_1 =  rk + alphak * matrizRala[2] * pk
        betak_1 = (rk_1.T * rk_1) / (rk.T * rk)
        betak_1 = betak_1[0,0]
        pk_1 = -rk_1 + betak_1 * pk
        xk, rk, pk = xk_1, rk_1, pk_1
    return xk

    #Implementación:
    #Definamos x0:
    x0 = [1 for i in range(len(matrizRala[0]))]

    print(gradiente_conjugado(x0, b, A = matrizRala))


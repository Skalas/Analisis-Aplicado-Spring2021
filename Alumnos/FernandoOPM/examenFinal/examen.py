import numpy as np
import random
# Nota el algoritmo de gradiente conjugado hace una prueba con un vector de dimension 10 y por eso estan alteradas
#las dimensiones de los vectores aleatorios
#incluyo algunos metodos que hice para la clase

def gradiente(f, x, h = .0001):
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        z = np.zeros(n)
        z[i] = h/2
        grad[i] = (f(x+z) - f(x-z))/h
    return grad

def hessiana(f, x, h = .0001):
    n = len(x)
    hess = np.zeros((n,n))
    for i in range(n):
        w = np.zeros(n)
        w[i] = h
        for j in range(n):
            if i==j:
                hess[i][j] = (-f(x+2*w) +16*f(x+w) - 30*f(x) + 16*f(x-w) -f(x-2*w))/(12*h**2)
            else:
                z = np.zeros(n)
                z[j] = h
                hess[i][j] = (f(x + w + z) - f(x - w + z) - f(x - z + w) + f(x - z - w))/(4*h**2)
    return hess

def condicionesnecesarias(f, x, h = .0001):
    resp = True
    n = len(x)
    grad = gradiente(f, x, h)
    for i in range(n):
        if abs(grad[i]) > h:
            resp = False
            break

    return resp        


def condicionesWolfe(f, x, p, a, c1=.5, c2=.9):
    resp = True
    if f(x + a*p) > f(x) + c1*a*np.dot(gradiente(f, x), p):
        resp=False
    if np.dot(gradiente(f, x + a*p), p) < c2*np.dot(gradiente(f, x), p):
        resp=False
    return resp

def generaAlpha(f, x, pk, c1=1e-4, c2 = 0.5, tol=1e-5):
    a = 1
    rho = 1/3
    gp = np.dot(gradiente(f,x), pk)
    while f(x + a*pk) > f(x) + c1*a*gp:
        a = a * rho
    return a

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

def actualizacion_DFP(yk, sk, hk):
    return hk- np.dot(np.dot(hk,yk), np.dot(yk, hk))/np.dot(np.dot(yk, hk), yk) + np.matmul(sk, sk.T)/np.dot(yk, sk)

def BFGS(f, x0, tol = .0001, maxIter = 1000):
    xk = x0
    hk = hessiana(f, x0, tol)
    gk = gradiente(f,x0, tol)
    k = 0
    sk = 10
    while np.linalg.norm(gk) > tol and np.linalg.norm(sk) > tol and k < maxIter:
        pk = -np.dot(hk, gk)
        alpha = generaAlpha(f, xk, pk)
        xk1 = xk + alpha * pk
        sk = xk1 - xk
        Gk1 = gradiente(f, xk1)
        yk = Gk1 - gk
        #hk = BFGS_Hk(yk, sk, hk)
        actualizacion_DFP(yk, sk, hk)
        k += 1
        xk = xk1
        gk = Gk1
    return xk1, k



def cuadrados(x):
    suma = 0
    for i in range(len(x)):
        suma = suma + x[i]**2
    return suma

x0 = [(-1)**i*10 for i in range(10) ]

print("con la funcion cuadrados y el punto inicial x0:")
print(x0)
print("el algoritmo bfgs con la actualizacion dfp nos da como resultado el minimo:")
x = BFGS(cuadrados, x0)
print(x[0])
print("en:")
print(x[1])
print("iteraciones")

########gradiente conjugado ################
print("gradiente conjugado")

random.seed(173122) #  Cambien a su propia clave
Diag_A = [random.randint(1,1000) for x in range(10)]
b = [random.randint(1,1000) for x in range(10)]


def dirConj(x0, A, b):
    n = len(x0)
    r = multi(A, x0) -b
    for i in range(n):
        alpha = -r[i]*(1/A[i])
        x0[i]= x0[i] + alpha
        r[i] = A[i]*x0[i] - b[i]
    """
    pk = []
    for i in range(n):
        z = np.zeros(n)
        z[i] = 1
        pk = pk + [z]
    for i in range(n):
        r = np.dot(A, x0) -b
        alpha = -np.dot(r, pk[i])/np.dot(np.dot(pk[i], A), pk[i])
        x0= x0 + alpha*pk[i]
    """
    return x0

def multi(A, x0):
    n = len(x0)
    x = np.zeros(n)
    for i in range(n):
        x[i] = A[i]* x0[i]
    return x
    


def gradConj(x0, A, b):
    rk = np.dot(A, x0) -b
    pk = -rk
    xk = x0
    cont =0
    while np.dot(rk,rk) >.0001 and cont<1000:
        ak = np.dot(rk, rk)/np.dot(np.dot(pk, A), pk)
        xk_1 = xk + ak*pk
        rk_1 = rk + ak*np.dot(A, pk)
        bk_1 = np.dot(rk_1, rk_1)/np.dot(rk, rk)
        pk_1 = -rk_1 + bk_1*pk
        xk = xk_1
        pk = pk_1
        rk = rk_1
        cont = cont + 1
    return xk

def gradConjPrecond(x0, A, b, M):
    rk = np.dot(A, x0) -b
    yk = np.linalg.solve(M, rk)
    pk = -yk
    xk = x0
    while np.dot(rk,rk) >.0001:
        print("rk: ")
        print(rk)
        ak = np.dot(rk, yk)/np.dot(np.dot(pk, A), pk)
        xk_1 = xk + ak*pk
        rk_1 = rk + ak*np.dot(A, pk)
        yk_1 = np.linalg.solve(M, rk_1)
        print("yk+1:")
        print(yk_1)
        bk_1 = np.dot(rk_1, yk_1)/np.dot(rk, yk)
        pk_1 = -yk_1 + bk_1*pk
        print(pk_1)
        yk = yk_1
        xk = xk_1
        pk = pk_1
        rk = rk_1
    return xk

print("Se utiliza una adaptacion del metodo de direcciones conjugadas ya que como trabajamos con una matriz diagoal los vecotras canonicos sirven como conjunto conjugado")
print("")
print("prueba con vectores aleatorios de dimension 10")
print("diagonal:")
print(Diag_A)
print("b:")
print(b)

x0 = np.ones(len(b))
x = dirConj(x0, Diag_A, b)
print("x:")
print(x)
print("Diag_A * x:")
print(multi(Diag_A, x))



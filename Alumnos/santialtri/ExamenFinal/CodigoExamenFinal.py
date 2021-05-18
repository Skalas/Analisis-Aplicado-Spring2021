import numpy as np 
from numpy import linalg as LA

######################################################################
##Primero copio todas las funciones que hemos hecho en el curso
######################################################################

def funcion_Rosenbrock(x,a=1,b=100):
    """
    Funcion de Rosenbrock, definida de R^2-->R, i.e. es una funcion real de dos variables
    """
    f = (a - x[0])**2 + b*(x[1] - x[0])**2
    return f

def cuadrados(x):
    resultado=0
    for i in range(len(x)):
        resultado+=x[i]**2
    return resultado

def Grad(f, x0, h=1e-6, i=-1):
    """
    Funcion que calcula el Grad de una funcion en un punto
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
            Grad[j]= (f(x0 + z) - f(x0 - z))/h
    return Grad

def Hess(f, x0, h=1e-4, method = "basic"):
    """
    Funcion que calcula la Hessiana  de una funcion en un punto. 
    f: funcion sobre la cual queremos calcular la hessiana.
    x0: Punto sobre el cual queremos hacer el calculo
    h: nivel de precision para hacer el calculo
    method: Metodo por el cual se quiere hacer puede ser: 'basic', 'grad', 'centered', 'gradCentered'
    """
    n = len(x0)
    Hess = np.matrix(np.zeros((n,n)))
    for i in range(n):
        for j in range(n):
            z_i = np.zeros(n)
            z_j = np.zeros(n)
            if j<= i :
                z_i[i] = h
                z_j[j] = h
                if method == "basic":
                    Hess[i,j] = ( f(x0 + z_j +z_i) - f(x0 + z_i ) - f(x0+z_j) +f(x0)) / (h**2)
                    Hess[j,i] = Hess[i,j]
                elif method == "grad":
                    Hess[i,j] = (Grad(f,x0+z_j,h,i) - Grad(f,x0,h,i) + \
                                Grad(f,x0+z_i,h,j) - Grad(f,x0,h,j))/(2*h)
                    Hess[j,i] = Hess[i,j]
                elif method == "centered":
                    if i==j:
                        Hess[i,j] = (-f(x0+2*z_i) + 16*f(x0+z_i) - 30*f(x0)+\
                                    16*f(x0-z_i) - f(x0-2*z_i))  / (12*h**2)
                        Hess[j,i] = Hess[i,j]
                    else :
                        Hess[i,j] = (f(x0+z_i+z_j) - f(x0 + z_i - z_j) - \
                                    f(x0 - z_i + z_j) + f(x0-z_i-z_j))/(4*h**2)
                        Hess[j,i] = Hess[i,j]
                elif method == "gradCentered":
                        Hess[i,j] = (Grad(f,x0+z_j,h)[i] - Grad(f, x0-z_j,h)[i] + \
                                    Grad(f,x0+z_i,h)[j] - Grad(f,x0-z_i,h)[j])/(4*h)
                        Hess[j,i] = Hess[i,j]
    return Hess

def f_o_c(f,x, tol=1e-12):
    """
    Funcion que calcula las condiciones de primer orden
    """
    grad = np.array(Grad(f,x))
    if np.dot(grad, grad) < tol:
        return True
    else :
        return False

def s_o_c(f, x0, tol=1e-15):
    """
    Nos dice si se cumple la condicion de segundo orden
    """
    hess = Hess(f, x0, tol)
    if np.all(np.linalg.eigvals(hess) > tol) :
        return True
    else :
        return False


def condiciones_wolfe(f, x0, alpha, pk, c1=1e-4, c2=1e-2, tol=1e-5):
    """
    Funcion que evalua las condiciones de wolfe para una alpha.
    f:  funcion que optimizamos
    x0: punto anterior un numpy.array
    alpha: valor que cumplira condiciones de wolfe.
    pk: direccion de decenso un numpy.array
    """
    def grad(alpha): return Grad(f, x0+alpha*pk, tol)
    def phi(alpha): return f(x0 + alpha*pk)  # Ojo que phi(0) = f(x0)
    def linea(alpha): return phi(0) + c1 * alpha * np.dot(g_x0, pk)
    g_x0 = grad(0)  # grad(0) = Grad(f,x0)
    cond_1 = linea(alpha) - phi(alpha) >= 0
    cond_2 = np.dot(grad(alpha), pk) - c2 * np.dot(g_x0, pk) >= 0
    return cond_1 and cond_2


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

def genera_alpha2(f, x0, pk, c1=1e-4, tol=1e-5):
   """
   Backtracking LS i.e. Algoritmo que encuentra una alpha que cumpla condiciones de wolfe. 
   """
   alpha, rho, c = 1, 4/5, c1
   while f(x0 + alpha*pk)>f(x0) + c*alpha*np.dot(Grad(f, x0),pk):
       alpha*=rho
   return alpha

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
        alphak = alphak[0,0]
        xk_1 = xk + alphak * pk
        rk_1 =  rk + alphak * A * pk
        betak_1 = (rk_1.T * rk_1) / (rk.T * rk)
        betak_1 = betak_1[0,0]
        pk_1 = -rk_1 + betak_1 * pk
        xk, rk, pk = xk_1, rk_1, pk_1
    return xk


def gradiente_conjugado2(x0, A, b):
    xk = x0
    b = np.matrix(b).T
    rk = np.dot(A, x0) - b
    pk = -rk
    while not (LA.norm(rk) ==  0):
        alphak = np.dot(rk,rk) / (np.transpose(pk) * A * pk)
        alphak = alphak[0,0]
        xk_1 = xk + alphak * pk
        rk_1 =  rk + alphak * A * pk
        betak_1 = (rk_1.T * rk_1) / (rk.T * rk)
        betak_1 = betak_1[0,0]
        pk_1 = -rk_1 + betak_1 * pk
        xk, rk, pk = xk_1, rk_1, pk_1
    return xk

def GC(A, b, x0):
    """
    Funcion de gradiente conjugado.
    """  
    r = b - np.dot(A, x0)
    p = r
    rsold = np.dot(np.transpose(r), r)
    xk=x0
    
    for i in range(len(b)):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(np.transpose(p), Ap)
        xk = xk + np.dot(alpha, p)
        r = r - np.dot(alpha, Ap)
        rsnew = np.dot(np.transpose(r), r)
        if np.any(np.sqrt(rsnew)) < .002:
            break
        p = r + (rsnew/rsold)*p
        rsold = rsnew
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

def DFP_Bk(yk, sk, Bk):
    """
    Funcion que calcula La actualizacion DFP de la matriz Bk
    In:
      yk: Vector n
      sk: Vector n
      Bk: Matriz nxn
    Out:
      Bk+1: Matriz nxn
    """
    n = len(yk)
    rhok = 1 / (yk.T*sk)
    Vk = (np.eye(n) - rhok * yk*sk.T)
    Bk1 = Vk * Bk * Vk + rhok * yk * yk.T
    return Bk1


def DFP_Hk1(yk, sk, Hk):
    """
    Funcion que calcula La actualizacion DFP de la matriz Hk
    In:
      yk: Vector n
      sk: Vector n
      Hk: Matriz nxn
    Out:
      Hk+1: Matriz nxn
    """
    #trato de hacer la Hk_1 de tres maneras distintas 
    Hk_1 = Hk - (np.dot(Hk, np.dot(yk, np.dot(yk, Hk))))/ (yk.T.dot(Hk).dot(yk)) + (sk.dot(sk.T)) / (yk.T.dot(sk))
    return Hk_1

def DFP_Hk2(yk, sk, Hk):
    """
    Funcion que calcula La actualizacion DFP de la matriz Hk
    In:
      yk: Vector n
      sk: Vector n
      Hk: Matriz nxn
    Out:
      Hk+1: Matriz nxn
    """
    #trato diferente de producir la Hk1
    Hk1 = Hk - (Hk * yk * yk.T * Hk)/(yk.T * Hk * yk) + (sk * sk.T)/(yk.T * sk)
    return Hk1

def DFP_Hk3(yk, sk, Hk):
    """
    Funcion que calcula La actualizacion DFP de la matriz Hk
    In:
      yk: Vector n
      sk: Vector n
      Hk: Matriz nxn
    Out:
      Hk+1: Matriz nxn
    """
    #trato diferente de producir la Hk1
    yk = np.array([yk]).T
    sk = np.array([sk])
    Hk1 = Hk - (Hk*(yk.dot(yk.T))* Hk)/((yk.T.dot(Hk)).dot(yk)) + (sk*sk.T)/(yk.T.dot(sk.T))
    return Hk1


def BFGS_Hk(yk, sk, Hk):
    """
    Funcion que calcula La actualizacion BFGS de la matriz Hk
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


def BFGS_Bk(yk, sk, Bk):
    """
    Funcion que calcula La actualizacion BFGS de la matriz Bk
    In:
      yk: Vector n
      sk: Vector n
      Bk: Matriz nxn
    Out:
      Bk+1: Matriz nxn
    """
    return Bk - (np.dot(Bk, np.dot(sk, np.dot(sk, Bk)))) / (np.dot(sk, np.dot(Bk, sk))) + np.dot(yk, yk) / np.dot(yk, sk)

def BFGS1(f, x0, tol, H0, maxiter=10000):
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
        #Hk = BFGS_Hk(yk, sk, Hk)
        Hk = DFP_Hk1(yk, sk, Hk) ##aqui modifico a que la actualizacion de Hk sea DFP y no BFGS
        k += 1
        xk = xk1
        Gk = Gk1
    return xk1, k

def BFGS2(f, x0, tol, H0, maxiter=10000):
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
        #Hk = BFGS_Hk(yk, sk, Hk)
        Hk = DFP_Hk2(yk, sk, Hk) ##aqui modifico a que la actualizacion de Hk sea DFP y no BFGS
        k += 1
        xk = xk1
        Gk = Gk1
    return xk1, k

def BFGS3(f, x0, tol, H0, maxiter=10000):
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
        #Hk = BFGS_Hk(yk, sk, Hk)
        Hk = DFP_Hk3(yk, sk, Hk) ##aqui modifico a que la actualizacion de Hk sea DFP y no BFGS
        k += 1
        xk = xk1
        Gk = Gk1
    return xk1, k



def BLS(a,f,xk,pk,h):         ## Backtracking Line Search
    rho=0.72
    c=h
    a0=a
    while f(xk +a0*pk) > f(xk) + c*a0*np.dot(np.transpose(Grad(f,xk,h)),pk):
        a0 = rho*a0
    return a0

def BFGS2(f,x0,h,a):
        n = len(x0)
        Bk = Hess(f,x0)
        gf = Grad(f,x0)
        Hk = LA.inv(Bk)
        while LA.norm(Grad(f,x0))>h:
            pk = -Hk*Grad(f,x0,h)
            xk = x0 + BLS(a,f,x0,pk,h)*pk
            sk = xk - x0
            yk = Grad(f,xk,h) - Grad(f,x0,h)
            rho = float(1/(np.transpose(yk)*sk))
            Hk1 = (np.identity(n)-rho*sk*np.transpose(yk)) * Hk * (np.identity(n)-rho*yk*np.transpose(sk)) + rho*sk*np.transpose(sk)
            x0 = xk
            Hk = Hk1
        return xk

######################################################################
#Aqui ya empieza mi respuesta a las preguntas del examen
######################################################################


############################# DFP ##########################

print('-----------------------')
x0 = [(-1)**i*10 for i in  range(10)]


H_0= np.identity(10) #empezare con la identidad para H0

print(BFGS1(cuadrados,x0, .002, H_0)) #hice tres modificaciones para el DFP de Hk, esta es BFGS1

print(BFGS2(cuadrados,x0, .002, H_0)) #hice tres modificaciones para el DFP de Hk, esta es BFGS2

print(BFGS3(cuadrados,x0, .002, H_0)) #hice tres modificaciones para el DFP de Hk, esta es BFGS3

#empezare ahora con un multiplo de la identidad para H0

H_00= 5*H_0

print(BFGS1(cuadrados,x0, .002, H_00)) 

print(BFGS2(cuadrados,x0, .002, H_00)) 

print(BFGS3(cuadrados,x0, .002, H_00))

############################# Gradiente Conjugado ##########################

#lo que nos piden se puede resolver con Scipy usando csr_matrix() y csc_matrix(), sin embargo no podemos usar Scipy ahora

print('------------------------')
import random 

random.seed(171886)

#verificamos que el algoritmo de GC funciona para ejemplo trival
n = 15
A=np.matrix(np.eye(n)).T
b= np.zeros(n) + 1
identidad15= np.identity(n)
x_0=np.matrix(np.zeros(n)).T

print(gradiente_conjugado_precond(x_0, A, b,identidad15 )) 
print('------------------------')



Diag_A = [random.randint(1,1000) for x in range(1000000)]
b = [random.randint(1,1000) for x in range(1000000)]

#b[i] ------> asi se obtiene la entrada i del vector b
#print(Diag_A[3]) asi obtengo la entrada i,i de la matriz diagonal A

def generaRala(x):
    """
    A partir de un vector generaremos una matriz diagonal
    In: el vector x sera justo la diagonal de la matriz A
    Out: matriz cuya diagonal es el vector x y las demas entradas son nulas
    """
    n= len(x)
    M= np.matrix(np.eye(n))
    for i in range(n):
        M[i,i] = x[i]
    
    return M

print(generaRala([2,4,3])) #vemos que funciona generaRala para ejemplo trivial

A= generaRala(Diag_A) #Ahora creamos la matriz A

n = 1000000
identidad= np.identity(n)
x_0=np.matrix(np.zeros(n)).T

print(gradiente_conjugado_precond(x_0, A, b,identidad )) #Resolvemos Ax=b con GC, aunque tarda mucho

#una vez que se trata de una matriz diagonal A, sabemos que el resultado x de Ax=b
# debe ser un vector cuya entrada i sea igual a la entrada i de b entre la entrada i,i de A
#es decir, x[i]= b[i]/A[i,i]

def ResuelveConADiagonal(A,b):
    """
    Funcion que da el resultado de x para Ax=b si A es diagonal
    in: A que es matriz diagonal
        b que es vector
    out: x vector solucion de Ax=b
    """
    n=len(b)
    x= np.zeros(n)
    for i in range(n):
        x[i]= b[i]/A[i,i]
    
    return x


print('El resultado de Ax=b es :')
print(ResuelveConADiagonal(A,b))

def ResuelveConADiagonal2(A,b):
    """
    Funcion que da el resultado de x para Ax=b si A es matriz diagonal
    in: A que es vector, se manda como vector en vez de matriz pues diag(A) es vector
        b que es vector
    out: x vector solucion de Ax=b
    """
    n=len(b)
    x= np.zeros(n)
    for i in range(n):
        x[i]= b[i]/A[i]
    
    return x

print('El resultado de Ax=b es :')
print(ResuelveConADiagonal2(Diag_A, b))














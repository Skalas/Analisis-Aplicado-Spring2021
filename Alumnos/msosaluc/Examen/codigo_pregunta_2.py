import numpy as np
import random
from numpy import linalg as la

#Utilizo los parametros a=1 y b=100 como menciona wikipedia del link que
#nos envio con el examen.

def Rosenbrock(x0):
        a=1
        b=100
        x=x0[0]
        y=x0[1]
        f = (a-x)**2 + b*(y-x**2)**2
        return f

#Funcion para calcular gradiente
def Gradiente(f, x0, h=1e-6):
    n = len(x0)
    G=np.zeros(n)
    for j in range(n):
        z = np.zeros(n)
        z[j] = h/2
        G[j]= (f(x0 + z) - f(x0 - z))/h
    return G

#Funcion para calcular Hessiana
def Hessiana(f,x0,h):
    n = len(x0)
    s= (n,n)
    H = np.zeros(s)
    for i in range(n):
        for j in range(n):
            z1 = np.zeros(n)
            z2 = z1
            z1[i] += h
            z2[j] += h
            x1 = x0 + z1 + z2
            x2 = x0 + z1 - z2
            x3 = x0 - z1 + z2
            x4 = x0 - z1 - z2
            H[i,j] = (f(x1) - f(x2) - f(x3) + f(x4))/(4*(h**2))
    return H


#Newton para minimizar tomando alpha=1
def alg_newton(f,xk,h,maxIt=10000):
    alpha = 1
    for k in range(maxIt):
        Bk = Hessiana(f,xk,h)
        pk = la.inv(Bk).dot(-Gradiente(f,xk,h))
        xk = xk + alpha*pk
    return xk

#Usamos BLS para buscar alpha
def BLS(a,f,xk,pk,h):         ##  Algoritmo 3.1: Backtracking Line Search
    p=random.uniform(0,1)
    c=random.uniform(0,1)
    a0=a
    while f(xk +a0*pk) > f(xk) + c*a0*np.dot(np.transpose(Gradiente(f,xk,h)),pk):
        a0 = p*a0
    return a0

def alg_newton_BL(f,xk,h,maxIt=10000):
    alpha = 1
    for k in range(maxIt):
        Bk = Hessiana(f,xk,h)
        pk = la.inv(Bk).dot(-Gradiente(f,xk,h))
        alpha = BLS(1,f,xk,pk,h)
        xk = xk + alpha*pk
    return xk



x0 = [-3,-4]
print("El gradiente está dado por:")
print(Gradiente(Rosenbrock,x0,0.00001))
print("La Matriz Hessiana esta dada por:")
print(Hessiana(Rosenbrock,x0,0.00001))
print("El resultado obtenido con Newton fue:")
print(alg_newton(Rosenbrock,x0,0.000001))
print("El resultado obtenido con la busqueda lineal de Newton2 fue:")
print(alg_newton_BL(Rosenbrock,x0,0.000001))

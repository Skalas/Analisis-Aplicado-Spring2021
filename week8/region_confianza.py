import numpy as np
from derivadas import Grad, Hess, cuadrados
import matplotlib.pyplot as plt

def obtener_derivadas(f, xk, gk='', Bk=''):
    if gk == '':
        gk = Grad(f, xk)
    if Bk == '':
        Bk = Hess(f, xk)
    return gk, Bk


def mk(f, xk, pk, gk='', Bk=''):
    gk, Bk = obtener_derivadas(f, xk, gk, Bk) 
    return f(xk) + np.dot(gk, pk) + np.dot(np.dot(pk,Bk), pk)/2


def punto_cauchy(f, xk, deltak, gk='', Bk=''):  
    tauk = 1
    gk, Bk = obtener_derivadas(f, xk, gk, Bk) 
    cuadratica = np.dot(np.dot(gk,Bk), gk)
    if  cuadratica > 0:
        mintau = np.dot(gk,gk)**(3/2)/(deltak*cuadratica) 
        if mintau <1:
            tauk = mintau
    return -tauk * deltak * gk / (np.dot(gk,gk)**(1/2))


def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)


if __name__ == "__main__":
    print(punto_cauchy(cuadrados, [0,1,1,0], 100))
    x = np.linspace(0, 5, 50)
    y = np.linspace(0, 5, 40)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    plt.contour(X, Y, Z, colors='black');

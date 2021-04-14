import numpy as np
import matplotlib.pyplot as plt
from Clase_22_Febrero import Grad,Hessiana 
from Clase_25_Enero import cuadrados
from Clase_10_Marzo import punto_cauchy

def obtener_derivadas(f,xk,gk='',Bk=''):
    if gk == '':
        gk = Grad(f, xk)
    if Bk == '':
        Bk = Hessiana(f, xk)
    return gk, Bk

def mk(f,xk,pk,gk='',Bk=''):
    gk, Bk = obtener_derivadas(f, xk, gk, Bk)
    return f(xk) + np.dot(gk,pk) + np.dot(np.dot(gk,Bk),gk)/2

def f(x,y):
    return np.sin(x)**10 + np.cos(2+ y * x) * np.cos(x)

print(punto_cauchy(cuadrados, [0,1,1,0], 1))
x = np.linspace(0,5,50)
y = np.linspace(0,5,40)
X, Y = np.meshgrid(x,y)
Z = f(X, Y)
plt.contour(X,Y,Z, colors = 'black')

#No acabamos de graficar, hay errores en el código 


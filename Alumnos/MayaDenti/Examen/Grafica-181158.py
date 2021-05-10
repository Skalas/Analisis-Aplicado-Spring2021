
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as cm
from numpy.lib.mixins import _inplace_binary_method

def Rosenbrock(x0):
    """
    La función Rosenbrock se define como f(x,y)=(a-x)^2+b(y-x^2)^2
    El mínimo global es en (a,a^2)
    Si queremos cambiar el a, b se cambian dentro de la función
    """
    a=1
    b=100
    x=x0[0]
    y=x0[1]
    fun = (a-x)**2 + b*(y-x**2)**2
    return fun

#Gráfica en 2 d sobre lo que esta pasando:
x=np.linspace(-2,2,100)
y=np.linspace(-1,4,100)
X,Y=np.meshgrid(x,y)
Z=Rosenbrock([X,Y])

fig,ax=plt.subplots()
plt.contour(X,Y,Z)
plt.show()

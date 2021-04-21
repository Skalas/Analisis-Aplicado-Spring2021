import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import line_search

#Definimos la función de Rosenbrock
def rosenbrock(x, y, a, b):
    return (a - x)**2 + b * (y - x**2)**2
#Definimos el gradiente de la función de Rosenbrock
def grad_rosenbrock(x, y, a, b):
    return [2 * (a - x) - 4 * b * (y - x**2) * x, 2 * b * (y - x**2)]

#Tomamos a = 1 y b = 100
a = 1
b = 100

#Graficamos la función
figRos = plt.figure(figsize = (12, 7))
axRos = figRos.gca(projection = '3d')

X = np.arange(-2, 2, 0.15)
Y = np.arange(-1, 3, 0.15)
X, Y = np.meshgrid(X, Y)
Z = rosenbrock(X, Y, a, b)

surf = axRos.plot_surface(X, Y, Z, cmap = cm.gist_heat_r, linewidth = 0, antialiased = False)
axRos.set_zlim(0, 200)
figRos.colorbar(surf, shrink = 0.5, aspect = 10)
plt.show()

#Le aplicamos búsqueda lineal. Este método está implementado en la librería de scipy
#Definimos el punto inicial
x0 = np.array([-1.4, 1.1])
#Dirección de descenso
pk = grad_rosenbrock(x0[0], x0[1], a, b)

line_search(rosenbrock(X, Y, a, b), grad_rosenbrock(X, Y, a, b), x0, pk)

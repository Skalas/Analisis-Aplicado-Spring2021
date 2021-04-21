import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d


def rosenbrock(x, a = 1, b = 1):
    return (a-x[0])**2 + b*(x[1] -x[0]**2)**2 

fig = plt.figure()
ax = plt.axes(projection='3d')

x = np.linspace(-1,2,1000)
y = np.linspace(-1,2,1000)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#ax.plot3D(x,y,z)
plt.show()
from bfgs import BFGS
from derivadas import cuadrados
import numpy as np

x_ini = [(-1)**i*10 for i in range(10) ]
tol = 1e-9

print("Corriendo pregunta de examen")
print("Esperamos mínimo en 0")
x, k = BFGS(cuadrados, x_ini, tol, np.eye(10))
print(f'Llegué a {x} en {k} iteraciones')

print("Que básicamente es 0 numérico")
print("Lo demostramos con isclose por entradas")
print(np.isclose(x, 0, tol))
print("Y en total")
print(np.all(np.isclose(x, 0, tol)))

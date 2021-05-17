from bfgs import BFGS
from derivadas import cuadrados
import numpy as np

x_ini = [(-1)**i*10 for i in range(10) ]

print("Corriendo pregunta de examen")
print("Esperamos mínimo en 0")
x, k = BFGS(cuadrados, x_ini, 1e-9, np.eye(10))
print(f'Llegué a {x} en {k} iteraciones')

print("Que básicamente es 0 numérico")
print(np.isclose(x, 0))

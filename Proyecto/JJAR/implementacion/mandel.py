import numpy as np
import numpy.linalg as ln
import seaborn as sns
from scipy.stats import gamma
import math

"""
Código para generar heatmap de convergencia.
ATENCIÓN: Si estás tratando de recrear la figura, revisa en region_confianza.py que el output de el
método sea el xk y el el array de puntos. No OptimizeResult.
"""

import matplotlib.pyplot as plt

from linesearch import BL
from funciones import score, OptimizeResult
import numdifftools as nd

from region_confianza import region_confianza
from icecream import ic


def obj(x):
    pdf = lambda x, y: gamma.pdf(x, a=7) * gamma.pdf(y, a=8)
    return -100 * score(4.8, pdf, x[0], x[1])


# Dibujando
X = np.arange(1, 10, step=0.25)
Y = np.arange(1, 10, step=0.25)

# Estructura para guardar resultados:
# primeras dos columnas para x, y. otra para iteraciones
pts = np.zeros((len(X) * len(Y), 3))

# Contador de reps
k = 0

print("Corriendo loop")

for row in range(len(X)):
    for col in range(len(Y)):
        print(f"Iteración {k}")
        x = X[row]
        y = Y[col]

        x0 = [x, y]

        opr = region_confianza(obj, nd.Gradient(obj), nd.Hessian(obj), x0, maxiter=9)

        if opr.success:
            print(f"El punto {x0} convergió")
            pts[k] = [x, y, opr.nit]
        else:
            pts[k] = [x, y, np.nan]

        k += 1

# Scatterplot
# sns.scatterplot(x=pts[:, 0], y=pts[:, 1], hue=pts[:, 2])

# Heatmap
rs = np.reshape(pts[:, 2], (-1, len(X)))
sns.heatmap(rs, cmap="Blues")

# Mostrando y guardando
plt.title("Regiones de convergencia del alg.")
plt.show()
plt.savefig("mandelb-alg.png")

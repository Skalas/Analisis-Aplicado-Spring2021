from scipy.stats import gamma
from linesearch import BL
from funciones import score

import numpy as np

from region_confianza import region_confianza
import numdifftools as nd


from icecream import ic
import matplotlib.pyplot as plt


def obj(x):
    pdf = lambda x, y: gamma.pdf(x, a=7) * gamma.pdf(y, a=8)
    return -100 * score(4.8, pdf, x[0], x[1])


x0 = [8.4, 8.4]


xk, pts = region_confianza(obj, nd.Gradient(obj), nd.Hessian(obj), x0)
ic(f"Resultado logrado {xk}")

## Graficando
X, Y = np.mgrid[0:13:0.1, 0:13:0.1]
Z = obj([X, Y])

fig, ax = plt.subplots()

# Contornos de nivel
cont1 = ax.contourf(X, Y, Z, cmap="Blues")
cont2 = ax.contour(X, Y, Z, colors="k")
ax.clabel(cont2, fontsize=12)

# Puntos que visitó el algoritmo
plt.plot(pts[:, 0], pts[:, 1], "*-", color="r")

plt.savefig("final.png")
plt.show()

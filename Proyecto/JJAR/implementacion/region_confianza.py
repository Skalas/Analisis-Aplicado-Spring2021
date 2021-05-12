from math import sqrt
from funciones import OptimizeResult

import numpy as np
from numpy import linalg as ln

import matplotlib.pyplot as plt
import seaborn as sns
from condiciones import is_min

from matplotlib import ticker


sns.set()


def step_cauchy(gk, Bk, delta):
    """
    Encuentra el paso de Cauchy para el método región de confianza

    Args:
        gk: vector np.array
        Bk: matriz np.array
        delta: Escalar radio de la región de confianza

    Returns:
        El paso óptimo según Cauchy
    """
    tau_k = 1

    # multi_dot conjuga como queremos y además multiplica en orden óptimo
    cuadratica = ln.multi_dot([gk, Bk, gk])
    g_norm = ln.norm(gk)

    if cuadratica <= 0:
        tau_k = 1
    else:
        tau_k = min(g_norm ** 3 / delta * cuadratica, 1)

    # Calculando punto de Cauchy
    factor = (tau_k * delta) / g_norm
    p_C = -1 * factor * gk

    return p_C


def step_dogleg(Hk, gk, Bk, delta):
    """
    Encuentra paso del método dogleg de acuerdo a Nocedal

    Args:
        Hk: Inversa de la matriz Bk
        gk: Gradiente de f en xk
        Bk: Hessiana en xk o aproximación simétrica
        delta: Radio de la región de confianza

    Returns: np.array con el paso óptimo
    """

    # Calculamos el full step
    pB = -np.dot(Hk, gk)
    norm_pB = ln.norm(pB)

    # Si full step está en la región, lo regresamos
    if norm_pB <= delta:
        return pB

    # Calculamos pU
    pU = -(np.dot(gk, gk) / ln.multi_dot([gk, Bk, gk])) * gk
    dot_pU = np.dot(pU, pU)
    norm_pU = ln.norm(pU)

    # Si pU se sale de la región, usamos el punto de intersección con la frontera
    if norm_pU >= delta:
        return delta * pU / norm_pU

    # De otra forma resolvemos el camino óptimo resolviendo la ecuación cuadrática como en Nocedal.
    # Es decir: ||pU + (tau - 1)(pB - pU)||^2 = delta^2
    # Cambiando la notación: || A + (tau - 1)(B)||^2 = delta^2
    # Usando la fórmula cuadrática
    B = pB - pU

    normsq_B = np.dot(B, B)
    pU_dot_B = np.dot(pU, B)

    fact = pU_dot_B ** 2 - normsq_B * (dot_pU - delta ** 2)
    tau = (-pU_dot_B + sqrt(fact)) / normsq_B

    # Eligiendo de acuerdo a Nocedal pg.74 problema 4.16
    return pU + tau * B


def region_confianza(
    func,
    jac,
    hess,
    x0,
    metodo="Dogleg",
    delta_0=1.0,
    delta_max=100.0,
    eta=0.15,
    maxiter=1000,
):
    """
    Implementación del algoritmo de optimización de región de 
    confianza de acuerdo a Nocedal.

    Args:
        func: Función objetivo.
        jac: Función del jacobiano de `func`.
        hess: Función de la matriz hessiana de `func`.
        x0: Punto de inicio para el algoritmo.
        metodo: String que indica el método para usar. El default es 
        Dogleg.
        delta_0: Radio inicial de la región de confianza.
        delta_max: Radio máximo de la región de confianza.
        eta: Parámetro eta.
        maxiter: Numero máximo de iteraciones permitidas.

    Return:
        xk: El punto óptimo obtenido por el algoritmo
        pts: np array con la información de los puntos visitados, en 
        qué iteración, entre otra información

    Nota: Se puede modificar el output descomentando la línea al final 
    de la función. Esto haría el output un OptimizeResult.
    """

    # Inicializando parámetos
    pts = np.zeros((2000, len(x0) + 2))
    xk = x0
    delta_k = delta_0
    k = 0

    # Loop principal
    while not is_min(jac, hess, xk) and k <= maxiter:
        gk = jac(xk)
        Bk = hess(xk)

        try:
            Hk = np.linalg.inv(Bk)
        except:
            print(f"Bk no invertible {Bk} en xk {xk}")
            break

        # Book-keeping para graficar
        pts[k] = np.append(xk, [k, np.exp(delta_k)])

        # Seleccionamos pk con el Punto de Cauchy o Dogleg
        if metodo == "Cauchy":
            pk = step_cauchy(gk, Bk, delta_k)
        elif metodo == "Dogleg":
            pk = step_dogleg(Hk, gk, Bk, delta_k)
        else:
            # Error fatal. El default debería ser "Dogleg"
            break

        # Reducción real
        red_re = func(xk) - func(xk + pk)
        # reducción esperada
        red_esp = -(np.dot(gk, pk) + 0.5 * ln.multi_dot([pk, Bk, pk]))

        # Rho.
        rho_k = red_re / red_esp

        # Evitando caso esquina
        if red_esp == 0.0:
            rho_k = 1e99
        else:
            rho_k = red_re / red_esp

        norm_pk = ln.norm(pk)

        # Rho es muy pequeño, reducimos el radio de la región de confianza
        if rho_k < 0.25:
            delta_k = 0.25 * norm_pk
        else:
            # Rho ~= 1 y pk está en la frontera de la región. Expandimos el radio
            if rho_k > 0.75 and norm_pk == delta_k:
                delta_k = min(2.0 * delta_k, delta_max)

        # Actualizando el paso
        if rho_k > eta:
            xk = xk + pk
        else:
            xk = xk

        k += 1

    # Lo siguiente da opción para el tipo de output esperado
    return xk, np.ma.masked_equal(pts, 0.0)
    # return OptimizeResult(xk, k <= maxiter, k)

"""
Implementación de las funciones objetivo y funciones de apoyo para el
problema principal de optimización
"""

import numpy as np
from scipy.integrate import quad


# DEPRECATED
def conv(z, fg):
    """
    Función de convolución naive. Calcula una integral de convolución entre f & g.
    Es decir la integral (f*g)(z) = int_{0}^{z} f(x) g(z-x) dx.
    Está pensada para vectorizarse, no para usarse por si sola.
    Obs: Esta es la convolución para f, g con soporte no-negativo

    Args:
        z: la variable sobre la cual se hace la convolución
        fg: Las funciones a convolutir (ya multiplicadas)

    Returns:
        El valor de la integral de la convolución para z
    """
    res, _ = quad(fg, 0, z, args=(z,))

    return res


# DEPRECATED
def convoluciona():
    """ Función pública que permite accesar a la versión vectorizada de la función convolución """
    return np.vectorize(conv)


# DEPRECATED
def stat_max(va, n):
    """
    Calcula el n-ésimo estadístico de orden para la variable aleatoria `va`. Es decir la
    distribución del máximo.

    Args:
        va: Instancia de la clase scipy.stats.rv_continuous

    Return:
        Función de densidad del estadístico de orden.
    """

    return lambda x: n * va.pdf(x) * va.cdf(x) ** (n - 1)


# DEPRECATED
def stat_min(va, n):
    """
    Calcula el primer estadístico de orden para la variable aleatoria `va`. Es decir la
    distribución del mínimo.

    Args:
        va: Instancia de la clase scipy.stats.rv_continuous

    Return:
        Función de densidad del estadístico de orden.
    """

    return lambda x: n * va.pdf(x) * (1 - va.cdf(x)) ** (n - 1)


def score(p1, pdf, x, y):
    """ Implementación de la función score
    Calcula las posibles calificaciones finales dados los datos,
    criterios de calificación, y calificaciones dadas, ponderado
    por la probabilidad de obtener dicha calificación.

    Args:
        p1: Calificación del primer parcial. Escalar
        pdf: Función de densidad de probabilidad conjunta de parcia y final

    Returns:
        Calificación * probabilidad de obtenerla
    """
    par1 = p1 * np.ones_like(x)
    pars = np.stack((par1, x))
    p_lo = np.min(pars, axis=0)
    p_s = np.sum(pars, axis=0) - p_lo
    return pdf(x, y) * (0.6 * p_lo + 0.1 * p_s + 0.3 * y)


class OptimizeResult:
    """ Representa el resultado del proceso de optimización

    Attributes:
        x_star: array_like
            Punto resultante de la optimización
        success: bool
            Si se optimizó bien
        nit: int
            Número de iteraciones del algoritmo
    """

    def __init__(self, x_star, success, nit) -> None:
        self.x_star = x_star
        self.success = success
        self.nit = nit

    def __repr__(self) -> str:
        if self.success:
            msg = f"Se optimizó exitosamente la función en {self.nit} iteraciones con resultado en {self.x_star}"
        else:
            msg = f"No se logró optimizar la función en {self.nit} iteraciones"

        return msg

"""
Re-implementaición de los métodos vistos en clase con la ayuda de numdifftools para ayudar a la
reproducibilidad y correctud.
"""

import numpy as np

import numpy.linalg as ln


def f_o_c(nabla, x, tol=1e-12):
    """
    Función que calcula las condiciones de primer orden
    """
    return np.dot(nabla(x), nabla(x)) < tol


def s_o_c(H, x0, tol=1e-15):
    """
    Función que calcula las condiciones de optimialidad de segundo orden
    """
    return np.all(ln.eigvals(H(x0)) > tol)


def is_min(nabla, H, x0, tol=1e-25):
    """
    Dado una función, la función is_min nos va a regresar True si es un mínimo,
    False si no podemos garantizar que es un mínimo
    """
    return f_o_c(nabla, x0) and s_o_c(H, x0, tol)

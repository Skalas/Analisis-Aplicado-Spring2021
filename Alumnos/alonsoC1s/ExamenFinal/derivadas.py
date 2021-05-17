import numpy as np


def Grad(f, x0, h=1e-6, i=-1):
    """
    Función que calcula el Grad de una función en un punto
    """
    n = len(x0)
    if i in range(n):
        z = np.zeros(n)
        z[i] = h/2
        Grad = (f(x0 + z) - f(x0 - z))/h
    else:
        Grad = np.zeros(n)
        for j in range(n):
            z = np.zeros(n)
            z[j] = h/2
            Grad[j] = (f(x0 + z) - f(x0 - z))/h
    return np.array(Grad)


def Hess(f, x0, h=1e-4, method="basic"):
    """
    Función que calcula la Hessiana  de una función en un punto.
    f: función sobre la cual queremos calcular la hessiana.
    x0: Punto sobre el cual queremos hacer el cálculo
    h: nivel de precisión para hacer el cálculo
    method: Método por el cual se quiere hacer puede ser:
             'basic', 'grad', 'centered', 'gradCentered'
    """
    n = len(x0)
    Hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            z_i = np.zeros(n)
            z_i[i] = h
            z_j = np.zeros(n)
            z_j[j] = h
            if method == "basic":
                Hess[i, j] = (f(x0 + z_j + z_i) - f(x0 + z_i) -
                              f(x0+z_j) + f(x0)) / (h**2)
            elif method == "grad":
                Hess[i, j] = (Grad(f, x0+z_j, h, i) - Grad(f, x0, h, i) +
                              Grad(f, x0+z_i, h, j) - Grad(f, x0, h, j))/(2*h)
            elif method == "centered":
                if i == j:
                    Hess[i, j] = (-f(x0+2*z_i) + 16*f(x0+z_i) - 30*f(x0) +
                                  16*f(x0-z_i) - f(x0-2*z_i)) / (12*h**2)
                else:
                    Hess[i, j] = (f(x0+z_i+z_j) - f(x0 + z_i - z_j) -
                                  f(x0 - z_i + z_j) + f(x0-z_i-z_j))/(4*h**2)
            elif method == "gradCentered":
                Hess[i, j] = (Grad(f, x0+z_j, h)[i] - Grad(f, x0-z_j, h)[i] +
                              Grad(f, x0+z_i, h)[j] - Grad(f, x0-z_i, h)[j])\
                               / (4 * h)
    return Hess


# NOTA: Implementación lenta y fea.
def cuadrados(x):
    resultado = 0
    for i in range(len(x)):
        resultado += x[i]**2
    return resultado


def rosenbrock(x, a=1, b=100):
    return (a-x[0])**2 + b * (x[1] - x[1]**2)**2


if __name__ == '__main__':
    print(Grad(rosenbrock, [0, 0], h=0.0000001))
    print(Hess(rosenbrock, [1, 1], method="basic", h=1e-7))

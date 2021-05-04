import numpy as np


def DFP_BK(yk, sk, Bk):
    """
    Función que calcula La actualización DFP de la matriz Bk
    In:
      yk: Vector n
      sk: Vector n
      Bk: Matriz nxn
    Out:
      Bk+1: Matriz nxn
    """
    n = len(yk)
    rhok = 1 / (yk.T*sk)
    Vk = (np.eye(n) - rhok * yk*sk.T)
    Bk1 = Vk * Bk * Vk.T + rhok * yk * yk.T
    return Bk1


def DFP_HK(yk, sk, Hk):
    """
    Función que calcula La actualización DFP de la matriz Hk
    In:
      yk: Vector n
      sk: Vector n
      Hk: Matriz nxn
    Out:
      Hk+1: Matriz nxn
    """
    Hk1 = Hk - (Hk * yk * yk.T * Hk)/(yk.T * Hk * yk) + (sk * sk.T)/(yk.T * sk)
    return Hk1


def BFGS_HK(yk, sk, Hk):
    """
    Función que calcula La actualización BFGS de la matriz Hk
    In:
      yk: Vector n
      sk: Vector n
      Hk: Matriz nxn
    Out:
      Hk+1: Matriz nxn
    """
    n = len(yk)
    rhok = 1 / (yk.T*sk)
    Vk = (np.eye(n) - rhok * yk * sk.T)
    Hk1 = Vk.T * Hk * Vk + rhok * sk * sk.T
    return Hk1

## BFGS Bk


if __name__ == "__main__":
    print("")

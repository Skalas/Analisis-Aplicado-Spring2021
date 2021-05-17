import numpy as np
import numpy.linalg as la

np.seterr('raise')

def DFP_Bk(yk, sk, Bk):
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


# La buena
def DFP_Hk(yk, sk, Hk):
    """
    Función que calcula La actualización DFP de la matriz Hk
    In:
      yk: Vector n
      sk: Vector n
      Hk: Matriz nxn
    Out:
      Hk+1: Matriz nxn
    """

    # Nos aseguramos de que tengan forma (n, 1) i.e no (n,) para que funcione multi_dot
    sk = np.reshape(sk, (len(sk), 1))
    yk = np.reshape(yk, (len(yk), 1))

    print(sk.shape)

    # Hk1 = Hk - (Hk * yk * yk.T * Hk)/(yk.T * Hk * yk) + (sk * sk.T)/(yk.T * sk)

    # Nota: multi_dot existe para no hacer esto pero bueno
    # Hk1 = Hk - (np.dot(Hk, np.dot(yk, np.dot(yk.T, Hk))))
    sum1 = la.multi_dot([Hk, yk, yk.T, Hk])
    div1 = (np.dot(yk.T, np.dot(Hk, yk)))
    sum2 = np.dot(sk, sk.T)
    div2 = np.dot(yk.T, sk)
    Hk1 = Hk - (sum1/div1) + (sum2/div2)

    return Hk1


def BFGS_Hk(yk, sk, Hk):
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
    yk = np.array([yk]).T
    sk = np.array([sk]).T
    rhok = 1 / yk.T.dot(sk)
    Vk = (np.eye(n) - rhok * yk.dot(sk.T))
    Hk1 = Vk.T * Hk * Vk + rhok * sk.dot(sk.T)
    return Hk1


def BFGS_Bk(yk, sk, Bk):
    """
    Función que calcula La actualización BFGS de la matriz Bk
    In:
      yk: Vector n
      sk: Vector n
      Bk: Matriz nxn
    Out:
      Bk+1: Matriz nxn
    """
    return Bk - (np.dot(Bk, np.dot(sk, np.dot(sk, Bk)))) / (np.dot(sk, np.dot(Bk, sk))) + np.dot(yk, yk) / np.dot(yk, sk)

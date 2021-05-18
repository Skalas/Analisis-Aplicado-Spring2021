import numpy as np


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



def DFP_Hk(yk,sk,Hk):
    """
    Función que calcula la actualización DFP de la matriz Hk
    In:
        yk: Vector n
        sk: Vector n
        Hk: Matriz nxn
    Out:
        Hk+1: Matriz nxn
    """
    yk = np.array([yk]).T
    sk = np.array([sk]).T
    Hk1 =  Hk - np.dot( np.dot( ( np.dot(Hk,yk) ),yk.T ),Hk ) / ( np.dot( np.dot(yk.T,Hk), yk) ) + ( np.dot(sk,sk.T) )/( np.dot(yk.T,sk))
    #Hk1 = Hk - (Hk* yk * yk.T * Hk )/( yk.T * Hk * yk) + (sk * sk.T )/( yk.T * sk)
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

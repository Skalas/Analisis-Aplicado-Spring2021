# -*- coding: utf-8 -*-
"""gc.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MRkJ6de8fPgaJHprIgX4McnEm3NNcKem
"""

import random
import numpy as np

random.seed(176357) #  Cambien a su propia clave
Diag_A = [random.randint(1,1000) for x in range(1000000)]
b = [random.randint(1,1000) for x in range(1000000)]

"""Función de matriz rala """

def MatrizRala(Diag_A):
  A = np.zeros(3000000).reshape((3, 1000000))
  for i in range(1000000):
      A[0][i] = Diag_A[i]
      A[1][i] = i
      A[2][i] = i
  return A

A = MatrizRala(Diag_A)

"""La función MatrizA nos permite obetener toda la información alojada en la matriz de 1000000 x 1000000 entradas. """

def MatrizA(A, i, j):
  if i == j:
     valor = A[0][i][i]
  else:
     valor = 0
  return valor

def generar_conjunto_canonico(n):
    A = np.matrix(np.eye(n)).T
    return [A[i].T for i in range(n)]


def gradiente_conjugado(x0, A, b):
    xk = x0
    b = np.matrix(b).T
    rk = np.dot(A, x0) - b
    pk = -rk
    while not (rk.T * rk ==  0):
        alphak = rk.T * rk / (pk.T * A * pk)
        alphak= alphak[0,0]
        xk_1 = xk + alphak * pk
        rk_1 =  rk + alphak * A * pk
        betak_1 = (rk_1.T * rk_1) / (rk.T * rk)
        betak_1 = betak_1[0,0]
        pk_1 = -rk_1 + betak_1 * pk
        xk, rk, pk = xk_1, rk_1, pk_1
    return xk

def gradiente_conjugado_precond(x0, A, b, M):
    xk = x0
    b = np.matrix(b).T
    rk = np.dot(A, x0) - b
    yk = np.linalg.solve(M, rk)
    pk = -yk 
    while not (rk.T * rk ==  0):
        alphak = rk.T * yk / (pk.T * A * pk)
        alphak= alphak[0,0]
        xk_1 = xk + alphak * pk
        rk_1 =  rk + alphak * A * pk
        yk_1 = np.linalg.solve(M, rk_1)
        betak_1 = (rk_1.T * rk_1) / (rk.T * rk)
        betak_1 = betak_1[0,0]
        pk_1 = -yk_1 + betak_1 * pk
        xk, rk, pk, yk  = xk_1, rk_1, pk_1, yk_1
    return xk



if __name__ == '__main__':
    random.seed(176357)
    n= 10000000
    A = MatrizRala(Diag_A)
    b = [random.randint(1,1000) for x in range(n)]
    x0 = np.matrix(np.zeros(n)).T
    print(gradiente_conjugado_precond(x0, A, b, A))
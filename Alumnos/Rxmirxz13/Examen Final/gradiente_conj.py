import numpy as np
from numpy import linalg as LA
import random


random.seed(170309) #  Cambien a su propia clave
Diag_A = [random.randint(1,1000) for x in range(1000000)]


#definimos función que nos genera matriz rala. La idea de esta función será guardar en la primera fila el número de fila en el que se encuentra el número distinto de cero
#en la segunda fila guardar el número de columna donde se encuentra el número !=0 y en la tercera fila guardar el valor distinto de cero en  cuestión.
#La función que se definirá será solo para hacer matrices diagonales ralas

def mat_rala_diag(diag):
    #creamos matriz
    n=len(diag)
    mat_rala=[[0 for i in range(n)] for j in range(0,3)]
    for i in range(n):
        mat_rala[0][i]=i
        mat_rala[1][i]=i
        mat_rala[2][i]=diag[i]
    return mat_rala    


#ahora haremos función que multiplica a nuestra matriz rala por un vector pues necesitamos esa operación para que nuestro código de gc corra bien 

def mult_rala_vec(rala, vector):
    #creamos vector de salida
    m=len(rala[2])
    l=[0 for i in range(m)]
    for i in range(m):
        l[i]=vector[i]*rala[2][i]
    l=np.array(l)
    return l


def gradiente_conjugado_ralo(x0, A, b):
    xk = x0
    b = np.array(b).T
    rk = mult_rala_vec(A, x0) - b
    rk=np.array([rk])
    pk = -rk
    while not (LA.norm(rk) ==  0):
        alphak = LA.norm(rk) / mult_rala_vec(pk, A).T.dot(pk)
        #alphak= alphak[0,0]
        xk_1 = xk + alphak * pk
        rk_1 =  rk + alphak * mult_rala_vec(A, pk)
        betak_1 = LA.norm(rk_1) / LA.norm(rk)
        #betak_1 = betak_1[0,0]
        pk_1 = -rk_1 + betak_1 * pk
        xk, rk, pk = xk_1, rk_1, pk_1
    return xk


n=len(Diag_A)
A = mat_rala_diag(Diag_A)
b = [random.randint(1,1000) for x in range(1000000)]




x0 = np.zeros(n).T

print(gradiente_conjugado_ralo(x0, A, b))
import numpy as np 
from Clase_22_Febrero import Grad,Hessiana 
from Clase_25_Enero import cuadrados

def mk(xk):
    """PINTAR CURVAS DE NIVEL DE MK"""

def punto_cauchy(f, xk, deltak, gk='', Bk=''):
    tauk = 1
    if gk == '':
        gk = Grad(f,xk)
    if Bk == '':
        Bk = Hessiana(f,xk)
    cuadratica = np.dot(np.dot(gk,Bk), gk)
    if cuadratica > 0:
        mintau = np.dot(gk,gk)**(3/2)/(deltak*cuadratica)
        if mintau < 1:
            tauk = mintau
    return -tauk * deltak * gk / (np.dot(gk,gk)**(1/2))

if __name__ == "__main__":
    print(punto_cauchy(cuadrados, [0,1,1,0], 1))



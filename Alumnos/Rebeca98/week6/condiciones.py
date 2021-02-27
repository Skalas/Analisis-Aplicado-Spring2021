from gradiente import gradiente, cuadrados, Hess
import numpy as np

def cond_primer_orden(f,x, tol=1e-12):
    """
    Función que calcula las condiciones necesarias
    """
    grad = np.array(gradiente(f,x))
    if np.dot(grad, grad) < tol:
        return True
    else :
        return False

def cond_segundo_orden(f,x0,tol=1e-12):
    """  funcion que calcula las condiciones de segundo orden suficientes"""
    hess = Hess(f,x0)

    if np.all(np.linalg.eigvals(hess) > tol): #todos los eigenvalores debe ser positivos
        return True
    else : 
        return False

x0 = [0,0,0,0] ##es el minimo 
x1 = [1,1,1,1]

print("\nCumple las condiciones de primer orden: ",cond_primer_orden(cuadrados, x1))
print("\nCumple las condiciones de segundo orden: ",cond_segundo_orden(cuadrados, x1))

def es_min(f,x0):
    if cond_primer_orden(f,x0) and cond_segundo_orden(f,x0):
        return True
    else:
        return False

print("\nEs minimo: ", es_min(cuadrados, x1))


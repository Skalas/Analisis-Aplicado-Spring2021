"""
Análisis Aplicado

IMPLEMENTACIÓN ALGORITMO BÚSQUEDA LINEAL

"""
import numpy as np
from LabClase3 import potencias
from LabClase5 import Grad, Hessiana
from LabClase6 import is_min
from LabClase7 import genera_alpha, is_pos_def, modificacion_hessiana

# IMPLEMENTAR LA FUNCIÓN DE COSTOS
# La función de Rosenbrok está definida como sigue: 
# f(x1,x2) = (1-x1)^2 + 100(x2-x1^2)^2
def rosenbrok(X):
    """
    Esta función regresa la función de Rosenbrok que va de R^2 a R
    dado X = (x1, x2)
    """
    if len(X) == 2:
        fr = (1-X[0])**2 + 100*(X[1]-X[0]**2)**2
        return fr
    else:
        return("Esta función recibe un vector de dos entradas")

def busqueda_lineal(f, x0, metodo = "max_desc"):
    """
    Esta función implementa el algoritmo de búsqueda lineal sobre f dado 
    el punto inicial de búsqueda x0. 
    El parámetro 'metodo' determina si la búsqueda se hará por máximo 
    descenso, por Newton o Quasi-Newton. 
    La elección entre estos últimos dos métodos depende de si la hessiana
    de f es positiva definida o si debe hacerse una modificación a ésta
    para obtener Bk positiva definida. 
    """
    xk = x0
    grad_fk = Grad(f, x0)
    hess_fk = Hessiana(f, x0)
    
    # Definir dirección de descenso pk
    while not is_min(f,xk):
        if metodo == "max_desc":
            n = len(grad_fk)
            I = np.identity(n)
            pk = np.dot(-I, grad_fk).tolist()

        elif metodo == "Newton/Quasi-Newton":
            if is_pos_def(hess_fk):
                inv_hess_fk = np.linalg.inv(hess_fk)
                pk = np.dot(- inv_hess_fk, grad_fk).tolist()
                pk = pk[0]

            else:
                Bk = modificacion_hessiana(hess_fk)
                inv_Bk = np.linalg.inv(Bk)
                pk = np.dot(- inv_Bk,grad_fk).tolist()
                pk = pk[0]

        # Definir tamaño del paso alphak
        xk = np.array(xk)
        pk = np.array(pk)
        alphak = genera_alpha(f, xk, pk)
        print(xk, '\n', alphak, '\n', pk)
        
        # Iterar
        xk_1 = xk + alphak * pk
        print(xk_1)
        
        xk = xk_1
        grad_fk = Grad(f, xk)
        hess_fk = Hessiana(f, xk) 
        
    x_min = xk
    return x_min

print(rosenbrok([4,4]))
print(rosenbrok([4,4,4]))

# Como podemos ver si intentamos implementar el algoritmo con Newton entramos
# en un loop :\
if __name__ == '__main__':
    x0 = [1,1]
    print(busqueda_lineal(rosenbrok, x0, "Newton/Quasi-Newton"))

# Ahora implementamos con Máximo Descenso
# Pensé que con este si iba a jalar pero también entro en un loop
# Entiendo que es por la función y no por el algoritmo porque si lo corro con
# otra función potencias si jala. Adjunto la prueba. 
# Ya no me dio tiempo para encontrar el error. 
if __name__ == '__main__':
    x0 = [1,1,1,1]
    print(busqueda_lineal(potencias, x0))
    print(busqueda_lineal(potencias, x0))
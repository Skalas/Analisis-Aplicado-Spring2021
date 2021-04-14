"""
Para abrir la carpeta APLICADO poner APL y luego tab (->|), poner git clone y código del URL para copiar repos. 
Para actualizar el repositorio:
    Ir al repo del profe y copiar el link --> abrimos terminal --> cd la carpeta 
    git remote add repo-clase link del profe. git pull repo-clase main. press i, write merge message, esc, :wq, enter.
    git push origin
Para crear un archivo: mkdir nombre del archivo 
Para copiar los archivos que acaben en .py dentro de la carpeta week7 dentro de la carpeta dos arriba, 
    dos arriba de donde estoy y copialos en el . que es donde estoy parada ahorita 
    cp ../../week7/*.py .
Para commitear un archivo: 
    git status, git add . , git status, git commit -m 'Mensaje que quiera'  
    OJO: Aquí modificamos el archivo del profe, no el nuestro. Este NO es el del pull request 
Para commitear en nuestro archivo:
    Guardamos el archivo en nuestra carpeta de alumnos dentro de Analisis-Aplicado-Spring2021
    git add nombre.py, git commit -m 'Mensaje', git status, git restore ../../week7/wolfe.py, git status, 
    git push origin main 
Nos pasamos al browser de git --> refresh --> pull request --> create pull request 

Solo estoy copiando lo del archivo wolfe.py en esta sesión, se trabajo también el 3 de marzo
"""
import numpy as np 
from Clase_22_Febrero import Grad,Hessiana 
from Clase_25_Enero import cuadrados

def condiciones_wolfe(f, x0, alpha, pk, c1=1e-4, c2 = 1e-2, tol=1e-5):
    """
    Función que evalúa las condiciones de wolfe para una alpha. 
    f:  función que optimizamos
    x0: punto anterior un numpy.array
    alpha: valor que cumplirá condiciones de wolfe. 
    pk: dirección de decenso un numpy.array
    """
    grad = lambda alpha: Grad(f,x0+alpha*pk, tol) #Las lambda functions definen una función dentro de una función
    phi = lambda alpha: f(x0 + alpha*pk) # Ojo que phi(0) = f(x0)
    linea = lambda alpha: phi(0) + c1 * alpha *np.dot( g_x0, pk)
    g_x0 = grad(0) # grad(0) = Grad(f,x0)
    cond_1 = linea(alpha) - phi(alpha) >= 0
    #print(phi(4))
    cond_2 = np.dot(grad(alpha), pk) - c2 * np.dot(g_x0, pk) >=0
    return  cond_1 and cond_2 


if __name__ == '__main__':
    print(condiciones_wolfe(cuadrados, np.array([1,1,1,1]), 1, np.array([-1,-1,-1,-1])))


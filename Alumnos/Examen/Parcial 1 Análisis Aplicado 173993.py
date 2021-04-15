import numpy as np

def rosenbrock(x,a=1,b=100):
    """
    Definición de la función de Rosenbrock
    """
    f=0

    f = (a-x[0])**2 + b*(x[1]-x[0]**2)**2
    return f

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
        Grad=np.zeros(n)
        for j in range(n):
            z = np.zeros(n)
            z[j] = h/2
            Grad[j]= (f(x0 + z) - f(x0 - z))/h
    return Grad


def Hess(f, x0, h=1e-4, method = "basic"):
    """
    Función que calcula la Hessiana  de una función en un punto. 
    f: función sobre la cual queremos calcular la hessiana.
    x0: Punto sobre el cual queremos hacer el cálculo
    h: nivel de precisión para hacer el cálculo
    method: Método por el cual se quiere hacer puede ser: 'basic', 'grad', 'centered', 'gradCentered'
    """
    n = len(x0)
    Hess = np.matrix(np.zeros((n,n)))
    for i in range(n):
        for j in range(n):
            z_i = np.zeros(n)
            z_i[i] = h
            z_j = np.zeros(n)
            z_j[j] = h
            if method == "basic":
                Hess[i,j] = ( f(x0 + z_j +z_i) - f(x0 + z_i ) - f(x0+z_j) +f(x0)) / (h**2)
            elif method == "grad":
                Hess[i,j] = (Grad(f,x0+z_j,h,i) - Grad(f,x0,h,i) + \
                             Grad(f,x0+z_i,h,j) - Grad(f,x0,h,j))/(2*h)
            elif method == "centered":
                if i==j:
                    Hess[i,j] = (-f(x0+2*z_i) + 16*f(x0+z_i) - 30*f(x0)+\
                                 16*f(x0-z_i) - f(x0-2*z_i))  / (12*h**2)
                else :
                    Hess[i,j] = (f(x0+z_i+z_j) - f(x0 + z_i - z_j) - \
                                 f(x0 - z_i + z_j) + f(x0-z_i-z_j))/(4*h**2)
            elif method == "gradCentered":
                    Hess[i,j] = (Grad(f,x0+z_j,h)[i] - Grad(f, x0-z_j,h)[i] + \
                                 Grad(f,x0+z_i,h)[j] - Grad(f,x0-z_i,h)[j])/(4*h)
    return Hess

def modificacion_hessiana(Hessiana, l = 0.5):
    while not is_pos_def(Hessiana):
        Hessiana = Hessiana + l*np.eye(len(Hessiana))
    return Hessiana

def f_o_c(f,x, tol=1e-12):
    """
    Función que calcula las condiciones de primer orden
    """
    grad = np.array(Grad(f,x))
    if np.dot(grad, grad) < tol:
        return True
    else :
        return False

def s_o_c(f, x0, tol=1e-15):
    """
    Inserten aqui código para condiciones de segundo orden 
    """
    hess = Hess(f, x0, tol)
    print(np.linalg.eigvals(hess))
    if np.all(np.linalg.eigvals(hess) > tol) :
        return True
    else :
        return False

def is_min(f, x0, tol=1e-25) :
    """
    #Dado una función, la función is_min nos va a regresar True si es un mínimo, 
    #False si no podemos garantizar que es un mínimo
    """
    if f_o_c(f, x0) and s_o_c(f, x0, tol) :
        return True
    else :
        return False


def busqueda_lineal1( f, x0): 
    """
    Algoritmo de búsqueda lineal con máximo descenso
    """
    xk = x0
    while not (f_o_c(f,xk) and s_o_c(f,xk)):
        grad= Grad(f,xk)
        pk = -grad
        alpha = genera_alpha(f,xk, pk)
        xk = xk + alpha*pk
    
    return xk 


#Suponemos alpha = 1, es decir, no cambia
def busqueda_lineal2( f, x0): 
    """
    Algoritmo de búsqueda lineal con dirección de Newton
    """
    xk = x0
    while not (f_o_c(f,xk) and s_o_c(f,xk)):
        grad = Grad(f,xk)
        hess = Hess(f,xk)
        pk = np.linalg.solve(hess, -grad)
        xk = xk + pk

    return xk 


#Vale la pena comparar el primer ejemplo, primero, por sí solo

#Búsqueda lineal
print(busqueda_lineal1(rosenbrock,[1,1]))
#print(busqueda_lineal1(rosenbrock,[1,0]))    #Con estos dos ejemplos tarda mucho en correr. 
#print(busqueda_lineal1(rosenbrock,[2,2]))

#Búsqueda de Newton
print(busqueda_lineal2(rosenbrock,[1,1]))
#print(busqueda_lineal2(rosenbrock,[1,0]))    #Los mismos ejemplos con Newton tardan mucho.
#print(busqueda_lineal2(rosenbrock,[2,2]))

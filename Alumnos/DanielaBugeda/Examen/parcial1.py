import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# ----------------------------------------
# Instanciación de función
# ----------------------------------------

a=1
b=100

def Rosenbrock(x):
    """
    Función objetivo que buscar optimizar. En este caso, encontrar su mínimo global.
    """
    rosen = (a-x[0])**2 + b*(x[1]-(x[0])**2)**2
    return rosen

def f(x,y):
    graph = (a-x)**2 + b*(y-(x)**2)**2
    return graph
# ----------------------------------------
# Preparación algoritmos de optimización
# ----------------------------------------

def gradiente(f, x0, h=1e-6, i=-1):
    """
    Función para calcular el gradiente de una función para un punto particular.
    """
    n = len(x0)
    if i in range(n):
        z = np.zeros(n)
        z[i]= h/2
        gradiente = (f(x0 + z) - f(x0 -z))/h
    else:
        gradiente = np.zeros(n)
        for j in range(n):
            z = np.zeros(n)
            z[j] = h/2
            gradiente[j] = (f(x0 + z) - f(x0 - z))/h
    return gradiente

def Hess(f, x0, h=1e-4, method="basic"):
    """
    Función para calcular la Hessiana de una función en un punto particular.
    f: función a la cual se le calculará la Hessiana
    x0: Punto sobre el cual se hará el cálculo
    h: nivel de precisión o tolerancia
    method: Método por el cual se calcula (basic, grad, centered o gradCentered)
    """
    n = len(x0)
    Hess = np.matrix(np.zeros((n,n)))
    for i in range(n):
        for j in range(n):
            z_i = np.zeros(n)
            z_j = np.zeros(n)
            if j<= i:
                z_i[i] = h
                z_j[j] = h
                if method == "basic":
                    Hess[i,j] = (f(x0 + z_j + z_i) - f(x0 + z_i) - f(x0 + z_j) + f(x0))/ (h**2)
                    Hess[j,i] = Hess[i,j]
                elif method == "grad":
                    Hess[i,j] = (gradiente(f, x0+z_j, h, i) - gradiente(f, x0, h, i) + gradiente(f, x0+z_i, h, j) - gradiente(f, x0, h, j))/(2*h)
                    Hess[j,i] = Hess[i,j]
                elif method == "centered":
                    if i==j:
                        Hess[i,j] = (-f(x0+2*z_i) + 16*f(x0+z_i) - 30*f(x0) + 16*f(x0-z_i) - f(x0-2*z_i))/(12*h**2)
                        Hess[j,i] = Hess[i,j]
                    else:
                        Hess[i,j] = (f(x0+z_i+z_j) - f(x0+z_i-z_j) - f(x0-z_i+z_j) + f(x0-z_j-z_j))/(4*h**2)
                        Hess[j,i] = Hess[i,j]
                elif method == "gradCentered":
                    Hess[i,j] = (gradiente(f, x0+z_j,h)[i] - gradiente(f, x0-z_j, h)[i] + gradiente(f, x0+z_i,h)[j] - gradiente(f, x0-z_i,h)[j])/(4*h)
                    Hess[j,i] = Hess[i,j]
    return Hess

def first_oc(f, x, tol=1e-12):
    """
    Función para calcular las condiciones de primer orden
    """
    grad = np.array(gradiente(f,x))
    if np.dot(grad, grad) < tol:
       return True
    else:
        return False

def second_oc(f, x, tol=1e-15):
    """
    Función para calcular las condiciones de segundo orden
    """
    hess = Hess(f,x,tol)
    if np.all(np.linalg.eigvals(hess) > tol):
        return True
    else:
        return False

def is_min(f, x, tol=1e-25):
    """
    Función que verifica si el punto ingresado es un mínimo o no para cierta función dada
    """
    if first_oc(f,x) and second_oc(f,x, tol):
        return True
    else: 
        return False

def cond_Wolfe(f, xk, alpha, pk, c1=1e-4, c2=1e-2, tol=1e-5):
    """
    Función para evaluar las condiciones de Wolfe en el paso alpha.
    f: función objetivo que buscamos optimizar
    xk: punto incial o punto anterior (numpy.array)
    alpha: valor para el cual buscamos evaluar las condiciones de Wolfe
    pk: dirección de decenso (numpy.array)
    """
    grad = lambda alpha: gradiente(f, xk+alpha*pk, tol)
    phi = lambda alpha: f(xk+alpha*pk) #phi(0) = f(xk)
    linea = lambda alpha: phi(0) + c1*alpha*np.dot(g_xk, pk)
    g_xk = grad(0) # = gradiente(f,xk)

    cond1 = linea(alpha) - phi(alpha) >= 0
    cond2 = np.dot(grad(alpha), pk) - c2 * np.dot(g_xk,pk) >= 0
    
    return cond1 and cond2

def genera_alpha(f, x0, pk, c1=1e-4, tol=1e-5):
    """
    Backtracking LS 
    (algoritmo para encontrar el alpha que cumpla las condiciones de Wolfe)
    """
    alpha, rho, c=1, 4/5, c1
    while f(x0+alpha*pk)>f(x0)+c*alpha*np.dot(gradiente(f,x0),pk):
        alpha*=rho
    return alpha

# ----------------------------------------
# Algoritmos Búsqueda Lineal y Nuewton
# ----------------------------------------

def busquedaLineal(f, x0, method = "MD" ):
    """
    Algoritmo de búsqueda lineal por dos métodos: 
    MD - utilizando la dirección de Máximo Descenso
    N - utilizando la dirección de Newton
    """
    xk = x0
    if method == "MD":
        while not (first_oc(f, xk) and second_oc(f, xk)):
            grad = gradiente(f, xk)
            pk = -grad
            alpha = genera_alpha(f,xk,pk)
            xk = xk + alpha*pk
    else:
        while not (first_oc(f, xk) and second_oc(f, xk)):
            grad = gradiente(f, xk)
            hess = Hess(f, xk)
            pk = np.linalg.solve(hess, -grad)
            #alpha = genera_alpha(f, x0,pk)
            alpha = 1
            xk = xk + alpha*pk
    
    return xk

# ----------------------------------------
# Pruebas
# ----------------------------------------

x0 = [2,2]

print(busquedaLineal(Rosenbrock, x0, "MD"))
print(busquedaLineal(Rosenbrock, x0, "N"))

# ----------------------------------------
# Gráfica
# ----------------------------------------

x = np.linspace(-2,2,50)
y = np.linspace(-1,3,50)
X, Y = np.meshgrid(x,y)
Z = f(X,Y)

grafica = plt.figure()
ejes = plt.axes(projection ='3d')
ejes.contour3D(X,Y,Z, 100, colors='red')
plt.show()
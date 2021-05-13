import numpy as np 

#-------------------------------------------------------------------
# Importamos las funciones del curso que utilizaremos en el proyecto
#--------------------------------------------------------------------

def Grad(f, x0, h=1e-6, i=-1):
    """
    Función que calcula el gradiente de una función en un punto. 
    In: 
        f - función sobre la cual queremos calcular el gradiente
        p - punto donde se calculará el gradiente
        h - tamaño del paso 
        i - iteración
    Out: 
        Grad - cálculo de gradiente en el punto
    """
    n = len(x0)
    if i in range(n):
        z = np.zeros(n)
        z[i] = h/2
        Grad = (f(x0 + z) - f(x0 - z))/h
    else:
        Grad = np.zeros(n)
        for j in range(n):
            z = np.zeros(n)
            z[j] = h/2
            Grad[j]= (f(x0 + z) - f(x0 - z))/h
    return Grad

def Hess(f, x0, h=1e-4, method = "basic"):
    """
    Función que calcula la Hessiana  de una función en un punto. 
    In:
        f - función sobre la cual queremos calcular la hessiana.
        x0 - punto sobre el cual queremos hacer el cálculo
        h - tamaño del paso, nivel de precisión para hacer el cálculo
        method - método por el cual se quiere hacer: 'basic', 'grad', 'centered', 'gradCentered'
    Out: 
        Hess - cálculo de la Hessiana en ese punto
    """
    n = len(x0)
    Hess = np.matrix(np.zeros((n,n)))
    for i in range(n):
        for j in range(n):
            z_i = np.zeros(n)
            z_j = np.zeros(n)
            if j<= i :
                z_i[i] = h
                z_j[j] = h
                if method == "basic":
                    Hess[i,j] = ( f(x0 + z_j +z_i) - f(x0 + z_i ) - f(x0+z_j) +f(x0)) / (h**2)
                    Hess[j,i] = Hess[i,j]
                elif method == "grad":
                    Hess[i,j] = (Grad(f,x0+z_j,h,i) - Grad(f,x0,h,i) + \
                                Grad(f,x0+z_i,h,j) - Grad(f,x0,h,j))/(2*h)
                    Hess[j,i] = Hess[i,j]
                elif method == "centered":
                    if i==j:
                        Hess[i,j] = (-f(x0+2*z_i) + 16*f(x0+z_i) - 30*f(x0)+\
                                    16*f(x0-z_i) - f(x0-2*z_i))  / (12*h**2)
                        Hess[j,i] = Hess[i,j]
                    else :
                        Hess[i,j] = (f(x0+z_i+z_j) - f(x0 + z_i - z_j) - \
                                    f(x0 - z_i + z_j) + f(x0-z_i-z_j))/(4*h**2)
                        Hess[j,i] = Hess[i,j]
                elif method == "gradCentered":
                        Hess[i,j] = (Grad(f,x0+z_j,h)[i] - Grad(f, x0-z_j,h)[i] + \
                                    Grad(f,x0+z_i,h)[j] - Grad(f,x0-z_i,h)[j])/(4*h)
                        Hess[j,i] = Hess[i,j]
    return Hess

def f_o_c(f,x, tol=.02):
    """
    Función que calcula las condiciones de primer orden.
    In: 
        f - función sobre la cual queremos probar las condiciones de primer orden
        x - punto donde estamos probando las condiciones
        tol - tolerancia (nivel de precisión)
    Out: 
        True/False - dependiendo de si se cumplen las condiciones de segundo orden
    """
    grad = np.array(Grad(f,x))
    if np.dot(grad, grad) < tol:
        return True
    else :
        return False

def s_o_c(f, x0, tol=.02):
    """
    Función que calcula las condiciones de segundo orden.
    In: 
        f - función sobre la cual queremos probar las condiciones de segundo orden
        x - punto donde estamos probando las condiciones
        tol - tolerancia (nivel de precisión)
    Out: 
        True/False - dependiendo de si se cumplen las condiciones de segundo orden
    """
    hess = Hess(f, x0, tol)
    if np.all(np.linalg.eigvals(hess) > tol) :
        return True
    else :
        return False

def condiciones_wolfe(f, x0, alpha, pk, c1=1e-4, c2 = 1e-2, tol=.02):
    """
    Función que evalúa las condiciones de wolfe para una alpha. 
    In: 
        f - función que buscamos optimizar
        x0 - punto inical 
        alpha - valor que cumplirá condiciones de wolfe
        pk - dirección de descenso
    Out:
        True/False - dependiento de si se cumplen las condiciones de Wolfe
    """
    grad = lambda alpha: Grad(f,x0+alpha*pk, tol)
    phi = lambda alpha: f(x0 + alpha*pk) # Ojo que phi(0) = f(x0)
    linea = lambda alpha: phi(0) + c1 * alpha *np.dot( g_x0, pk)
    g_x0 = grad(0) # grad(0) = Grad(f,x0)
    cond_1 = linea(alpha) - phi(alpha) >= 0
    cond_2 = np.dot(grad(alpha), pk) - c2 * np.dot(g_x0, pk) >=0
    return  cond_1 and cond_2 

def genera_alpha(f, x0, pk, c1=1e-4, c2 = 0.5, tol=.02):
    """
    Backtracking LS i.e. Algoritmo que encuentra un alpha que cumpla condiciones de wolfe.
    In:
        f - función que buscamos optimizar
        x0 - punto inicial
        pk - dirección de descenso
        c1 y c2 - constantes de Wolfe
        tol - tolerancia (nivel de precisión)
    Out: 
        alpha - devuelve el alpha óptimo que cumpla Wolfe
    """
    alpha, rho = 1, 3/4
    Gkpk = Grad(f, x0).dot(pk)
    while f(x0 + alpha*pk) > f(x0) + c1*alpha*Gkpk:
        alpha *= rho
    return alpha

#-----------------------------------------------------------------------
# Métodos numéricos de optimización (Búsqueda Lineal y Máximo Descenso)
#-----------------------------------------------------------------------

def BLNewton(f, x0, tol=.02, maxiter = 200): 
    """
    Algoritmo de optimización mediante el Método de Búsqueda Lineal con dirección de Newton. 
    In: 
        f - función a optimizar
        x0 - punto inicial
        tol - tolerancia 
    Out:
        xk - punto óptimo 
    """
    xk = x0
    i = 0
    grad = Grad(f,xk)
    hess = Hess(f,xk)
    while np.all(np.linalg.eigvals(hess) < tol) or (np.dot(grad, grad) > tol):
        grad = Grad(f,xk)
        hess = Hess(f,xk)
        pk = np.linalg.solve(hess, -grad)
        alpha = genera_alpha(f, xk, pk)
        """
        Si queremos ver cómo van cambiando las alphas y qué puntos visitamos quitar comillas
        print(alpha)
        print(xk)
        """
        xk = xk + alpha*pk

        if i == maxiter:
            break
        i = i+1
    return xk

def Maximo_Descenso(f, x0, tol = .02): 
    """
    Algoritmo de optimización mediante el Método de Máximo Descenso.
    In: 
        f - función a optimizar
        x0 - punto inicial
    Out:
        xk - punto óptimo 
    """
    xk = x0
    grad = Grad(f,xk)
    hess = Hess(f,xk)
    while np.all(np.linalg.eigvals(hess) < tol) or (np.dot(grad, grad) > tol):
        grad = Grad(f,xk)
        hess = Hess(f,xk)
        pk = -grad
        alpha = genera_alpha(f,xk,pk)
        xk = xk + alpha*pk

    return xk

#--------------------------------------------------------------
# Ejemplos económicos de optimización 
#--------------------------------------------------------------


def LagrangeanoCobbDouglas(x, alpha = 0.5, beta = 0.5, I = 50, p = [10,15], A = 1 ):
    """
    Maximizacion de utilidad Cobb-Douglas a partir de su Lagrangiano.
    In:
        x - vector de bienes y multiplicador
        p - vector de precios
        I - ingreso del individuo
        alpha - preferencia por bien x
        beta - preferencia por bin y
        A - constante de escalamiento
    Out:
        -L - Langrangeano (negativo pues queremos maximizar y el algoritmo minimiza)
    """
    L = A*(x[0]**alpha)*(x[1]**beta) + x[2]*(I-p[0]*x[0]-p[1]*x[1])
    return -L
    

def LagrangeanoCES(x, A=1, a=0.5, r=2, I=50, p=[10,15]):
    """
    Maximización de utilizad Función CES a partir de su Lagrangiano. 
    In:
        x - vector de bienes y multiplicador
        p - vector de precios
        a - parámetro de proporción
        r - elasticidad de sustitución
        I - ingreso del individuo
        A - constante de escalamiento
    Out:
        -L - Langrangeano (negativo pues queremos maximizar y el algoritmo minimiza)
    """
    b= 1-a
    L = A*(a*(x[0])**r + b*(x[1])**r)**(1/r) + x[2]*(I-p[0]*x[0]-p[1]*x[1])
    return -L


def Costos(x,c = [2,4], q_barra = 10, alpha = 0.5, beta = 0.5):
    """
    Maximización de una función de costos restringida a una producción media. 
    In:
        x - vector de trabajo y capital
        c - costo de factores
        q_barra - cantidad mínima de producción
        alpha - elasticidad por capital
        beta - elasticidad por trabajo
    Out:
        L - Langrangeano 
    """
    L = x[0]*c[0]+ x[1]*c[1] - x[2]*((x[0]**alpha)*(x[1]**(beta)) - q_barra)
    return L

#-----------------------------------------
# Pruebas
#-----------------------------------------

# Función Cobb-Douglas
#print(BLNewton(LagrangeanoCobbDouglas,[9,10,2])) 
#print(Maximo_Descenso(LagrangeanoCobbDouglas,[2.5,1.6,.0408])) 

# Función CES
print(BLNewton(LagrangeanoCES,[2,5,3]))
#print(Maximo_Descenso(LagrangeanoCES,[2,5,3]))

# Función Costos 
#c = [2,4], q_barra = 10, alpha = 0.5, beta = 0.5
#print(BLNewton(Costos,[15,11,3]))
#print(Maximo_Descenso(Costos,[15,11,3]))

#c = [2,2], q_barra = 5, alpha = 0.3, beta = 0.7
#print(BLNewton(Costos,[5,5,3]))
#print(Maximo_Descenso(Costos,[5,5,3]))



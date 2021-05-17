import numpy as np
import matplotlib.pyplot as plt

def Grad(f, x0, h=1e-6, i=-1):
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




def punto_cauchy(f, xk, deltak, gk='', Bk=''):
    tauk = 1
    if gk == '':
        gk = Grad(f,xk)
    if Bk == '':
        Bk = Hess(f,xk)
    cuadratica = np.dot(np.dot(gk,Bk), gk)
    if  cuadratica > 0:
        mintau = np.dot(gk,gk)**(3/2)/(deltak*cuadratica) 
        if mintau <1:
            tauk = mintau
    return -tauk * deltak * gk / (np.dot(gk,gk)**(1/2))


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
    Dado una función, la función is_min nos va a regresar True si es un mínimo, 
    False si no podemos garantizar que es un mínimo
    """
    if f_o_c(f, x0) and s_o_c(f, x0, tol) :
        return True
    else :
        return False

def condiciones_wolfe(f, x0, alpha, pk, c1=1e-4, c2 = 1e-2, tol=1e-5):
    """
    Función que evalúa las condiciones de wolfe para una alpha. 
    f:  función que optimizamos
    x0: punto anterior un numpy.array
    alpha: valor que cumplirá condiciones de wolfe. 
    pk: dirección de decenso un numpy.array
    """
    grad = lambda alpha: Grad(f,x0+alpha*pk, tol)
    phi = lambda alpha: f(x0 + alpha*pk) # Ojo que phi(0) = f(x0)
    linea = lambda alpha: phi(0) + c1 * alpha *np.dot( g_x0, pk)
    g_x0 = grad(0) # grad(0) = Grad(f,x0)
    cond_1 = linea(alpha) - phi(alpha) >= 0
    cond_2 = np.dot(grad(alpha), pk) - c2 * np.dot(g_x0, pk) >=0
    return  cond_1 and cond_2 


def genera_alpha(f, x0, pk, c1=1e-4, tol=1e-5):
    """
    Backtracking LS i.e. Algoritmo que encuentra una alpha que cumpla condiciones de wolfe. 
    """
    alpha, rho, c = 1, 4/5, c1
    while f(x0 + alpha*pk)>f(x0) + c*alpha*np.dot(Grad(f, x0),pk):
        alpha*=rho
    return alpha


def is_pos_def(Hessiana):
    """ A Rellenar """
    return True


def modificacion_hessiana(Hessiana, l = 0.5):
    while not is_pos_def(Hessiana):
        Hessiana = Hessiana + l*np.eye(len(Hessiana))
    return Hessiana


def obtener_derivadas(f, xk, gk='', Bk=''):
    if gk == '':
        gk = Grad(f, xk)
    if Bk == '':
        Bk = Hess(f, xk)
    return gk, Bk


def mk(f, xk, pk, gk='', Bk=''):
    gk, Bk = obtener_derivadas(f, xk, gk, Bk) 
    return f(xk) + np.dot(gk, pk) + np.dot(np.dot(pk,Bk), pk)/2


def punto_cauchy(f, xk, deltak, gk='', Bk=''):  
    tauk = 1
    gk, Bk = obtener_derivadas(f, xk, gk, Bk) 
    cuadratica = np.dot(np.dot(gk,Bk), gk)
    if  cuadratica > 0:
        mintau = np.dot(gk,gk)**(3/2)/(deltak*cuadratica) 
        if mintau <1:
            tauk = mintau
    return -tauk * deltak * gk / (np.dot(gk,gk)**(1/2))


def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

def cuadrados(x):
    resultado=0
    for i in range(len(x)):
        resultado += x[i]**4
    return resultado


def line_search(f, x0, method = "Maximo descenso"):
    xk = x0
    if method == "Newton":
        while not (f_o_c(f, xk) and s_o_c(f, xk)):
            grad = Grad(f, xk)
            hess = Hess(f, xk)
            pk = np.linalg.solve(hess, -grad)
            alpha = genera_alpha(f, xk, pk)
            xk = xk alpha*pk
    else:
        while not (f_o_c(f, xk) and s_o_c(f, xk)):
            grad = Grad(f, xk)
            pk = -grad
            alpha = genera_alpha(f, xk, pk)
            xk = xk + alpha*pk
    return xk
  


 # ST

def gradiente(f,xk):
    n = xk.size
    res = np.zeros(n)
    for i in range(n):
        res[i] = util.derivada_parcial(f,xk,i)
    return res

def backtracking_line_search(f, xk, p, c = .001, ro = .5):
    alpha = 1
    f_k = f(xk)
    gr = util.gradiente(f,xk)
    while(f(xk + alpha*p) <= f_k + c*alpha*gr.dot(p)):
        alpha*= ro
    return alpha


def make_pos_def(H):
    n = H.shape[0]
    Id = np.identity(n)
    t = .0001
    while(not util.is_pos_def(H)):
        H = H + t*Id
        t *= 2
    return H
def line_search_newton(f, x):
    eps = 0.0001
    n = 1000
    for i in range(n):
        B = make_pos_def(util.hessiana(f,x))
        gr = gradiente(f,x)
        p = np.linalg.solve(B, -1*gr)
        alpha = backtracking_line_search(f,x,p)
        x += alpha*p
    return x


import numpy as np
import pandas as pd

# Posición de las casas
'''
data = pd.read_csv('localidad.csv')
data.head()
df=pd.DataFrame(data)
d_x = df[df.columns[1]] # latitud (eje x) de las casas
d_x = d_x.to_numpy()
d_y = df[df.columns[2]] # longitud (eje y) de las casas
d_y = d_y.to_numpy()
'''

d_x = np.random.uniform(0,100,20)
d_y = np.random.uniform(0,178,20)


# Vector inicial con distribución uniforme del número de camiones que queremos
min_lat= np.amin(d_x)
max_lat = np.amax(d_x)
min_long = np.amin(d_y)
max_long = np.amax(d_y)
c_x = np.random.uniform(low=min_lat, high=max_lat, size=(10,)) # latitud (eje x) de los camiones
c_y = np.random.uniform(low=min_long, high=max_long, size=(10,)) # longitud (eje y) de los camiones


# Checo la distancia de cada casa al camión más cercano
def dist_casa_camion(casa_x, casa_y, camiones_x, camiones_y):

    n = camiones_x.size
    aux = 9999999999999999999999

    for i in range(n):
        dist = np.sqrt((casa_x - camiones_x[i])**2 + (casa_y - camiones_y[i])**2)
        if dist < aux:
            aux = dist

    return aux


# Encuentro la distancia total que van a recorrer todos los camiones para recoger la basura
# Esta es mi función objetivo
def distancia_total(casas_x, casas_y, camiones_x, camiones_y):

    k = casas_x.size
    total = 0

    for i in range(k):
        total = total + dist_casa_camion(casas_x[i], casas_y[i], camiones_x, camiones_y)

    return total


# Calculo el gradiente de mi función en algún punto
def gradiente(f, x0, h=1e-6, i=-1):

    m = len(x0)
    if i in range(n):
        z = np.zeros(n)
        z[i] = h/2
        gradiente = (f(x0 + z) - f(x0 - z))/h
    else:
        gradiente = np.zeros(n)
        for j in range(n):
            z = np.zeros(n)
            z[j] = h/2
            gradiente[j]= (f(x0 + z) - f(x0 - z))/h
    return gradiente


# Calculo la Hessiana de mi función en algún punto
def Hess(f, x0, h=1e-4, method = "basic"):
    
    l = len(x0)
    Hess = np.matrix(np.zeros((l,l)))
    for i in range(l):
        for j in range(l):
            z_i = np.zeros(l)
            z_j = np.zeros(l)
            if j<= i :
                z_i[i] = h
                z_j[j] = h
                if method == "basic":
                    Hess[i,j] = ( f(x0 + z_j +z_i) - f(x0 + z_i ) - f(x0+z_j) +f(x0)) / (h**2)
                    Hess[j,i] = Hess[i,j]
                elif method == "grad":
                    Hess[i,j] = (gradiente(f,x0+z_j,h,i) - gradiente(f,x0,h,i) + \
                                gradiente(f,x0+z_i,h,j) - gradiente(f,x0,h,j))/(2*h)
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
                        Hess[i,j] = (gradiente(f,x0+z_j,h)[i] - gradiente(f, x0-z_j,h)[i] + \
                                    gradiente(f,x0+z_i,h)[j] - gradiente(f,x0-z_i,h)[j])/(4*h)
                        Hess[j,i] = Hess[i,j]
    return Hess


# Calculo las condiciones de primer orden
def f_o_c(f, x0, tol = 1e-12):

    grad = np.array(gradiente(f, x0))
    if np.dot(grad, grad) < tol:
        return True
    else :
        return False


# Calculo las condiciones de segundo orden
def s_o_c(f, x0, tol = 1e-15):

    hess = Hess(f, x0, tol)
    if np.all(np.linalg.eigvals(hess) > tol) :
        return True
    else :
        return False


# Evalúo las condiciones de Wolfe en algún alpha
def condiciones_wolfe(f, x0, alpha, pk, c1=1e-4, c2 = 1e-2, tol=1e-5):

    grad = lambda alpha: gradiente(f,x0+alpha*pk, tol)
    phi = lambda alpha: f(x0 + alpha*pk) # Ojo que phi(0) = f(x0)
    linea = lambda alpha: phi(0) + c1 * alpha *np.dot( g_x0, pk)
    g_x0 = grad(0) # grad(0) = gradiente(f, x0)
    cond_1 = linea(alpha) - phi(alpha) >= 0
    cond_2 = np.dot(grad(alpha), pk) - c2 * np.dot(g_x0, pk) >=0
    return  cond_1 and cond_2 


# Encuentro la alpha que va a cumplir las condiciones de Wolfe
def genera_alpha(f, x0, pk, c1=1e-4, tol=1e-5):

    alpha, rho, c = 1, 4/5, c1
    while f(x0 + alpha*pk)>f(x0) + c*alpha*np.dot(Grad(f, x0),pk):
        alpha*=rho
    return alpha


# Implementamos el método de máximo descenso para encontrar la mejor red de distribución de camiones
def max_des(f, x0):

    xk = x0

    while not (f_o_c(f, xk) and s_o_c(f, xk)):
        grad = gradiente(f, xk)
        pk = -grad
        alpha = genera_alpha(f, xk, pk)
        xk = xk + alpha*pk
    
    return xk

print(max_des(distancia_total, (d_x,d_y,c_x,c_y)))

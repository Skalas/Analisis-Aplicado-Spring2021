import numpy as np
#import pandas as pd
import time

start_time = time.time()

# Posición de las casas
'''
data = pd.read_csv('localidad.csv')
data.head()
df=pd.DataFrame(data)
ca_x = df[df.columns[1]] # latitud (eje x) de las casas
ca_x = ca_x.to_numpy()
ca_y = df[df.columns[2]] # longitud (eje y) de las casas
ca_y = ca_y.to_numpy()
'''

ca_x = np.random.uniform(0, 100, 5)
ca_y = np.random.uniform(0, 178, 5)


# Vector inicial con distribución uniforme del número de camiones que queremos

n = 2 # número de camiones que se tienen disponibles

min_lat= np.amin(ca_x)
max_lat = np.amax(ca_x)
min_long = np.amin(ca_y)
max_long = np.amax(ca_y)
c_x = np.random.uniform(low = min_lat, high = max_lat, size = (n,)) # latitud (eje x) de los camiones
c_y = np.random.uniform(low = min_long, high = max_long, size = (n,)) # longitud (eje y) de los camiones

camiones = np.zeros(2*n)

for i in range(camiones.size):
    if i < n:
        camiones[i] = c_x[i]
    else:
        camiones[i] = c_y[i-n]


# Checo la distancia de cada casa al camión más cercano
def dist_casa_camion(casa_x, casa_y, camiones_coord):

    n = len(camiones_coord)
    k = int(n/2)
    aux = 999999

    for i in range(k):
        dist = np.sqrt((casa_x - camiones_coord[i-1])**2 + (casa_y - camiones_coord[i-1+k])**2)
        if dist < aux:
            aux = dist

    return aux


# Encuentro la distancia total que van a recorrer todos los camiones para recoger la basura
# Esta es mi función objetivo
def distancia_total(casas_x, casas_y, camiones_coord):

    k = casas_x.size
    total = 0

    for i in range(k):
        total = total + dist_casa_camion(casas_x[i], casas_y[i], camiones_coord)

    return total


# Calculo el gradiente de mi función en algún punto
def gradiente(f, casas_x, casas_y, x0, h = 1e-6, i = -1):

    m = len(x0)
    if i in range(m):
        z = np.zeros(m)
        z[i] = h/2
        gradiente = (f(casas_x, casas_y, x0 + z) - f(casas_x, casas_y, x0 - z))/h
    else:
        gradiente = np.zeros(m)
        for j in range(m):
            z = np.zeros(m)
            z[j] = h/2
            gradiente[j]= (f(casas_x, casas_y, x0 + z) - f(casas_x, casas_y, x0 - z))/h
    return gradiente


# Calculo la Hessiana de mi función en algún punto
def Hess(f, casas_x, casas_y, x0, h = 1e-4):
    
    n = len(x0)
    Hess = np.matrix(np.zeros((n,n)))
    for i in range(n):
        for j in range(n):
            z_i = np.zeros(n)
            z_j = np.zeros(n)
            if j <= i :
                z_i[i] = h
                z_j[j] = h

                Hess[i,j] = ( f(casas_x, casas_y, x0 + z_j + z_i) - f(casas_x, casas_y, x0 + z_i ) - f(casas_x, casas_y, x0 + z_j) + f(casas_x, casas_y, x0)) / (h**2)
                Hess[j,i] = Hess[i,j]
                
    return Hess


# Calculo las condiciones de primer orden
def f_o_c(f, casas_x, casas_y, x0, tol = 1e-12):

    grad = np.array(gradiente(f, casas_x, casas_y, x0))
    if np.dot(grad, grad) < tol:
        return True
    else :
        return False


# Calculo las condiciones de segundo orden
def s_o_c(f, casas_x, casas_y, x0, tol = 1e-15):

    hess = Hess(f, casas_x, casas_y, x0, tol)
    if np.all(np.linalg.eigvals(hess) > tol) :
        return True
    else :
        return False


# Evalúo las condiciones de Wolfe en algún alpha
def condiciones_wolfe(f, casas_x, casas_y, x0, alpha, pk, c1 = 1e-4, c2 = 1e-2, tol = 1e-5):

    grad = lambda alpha: gradiente(f, casas_x, casas_y, x0 + alpha * pk, tol)
    phi = lambda alpha: f(casas_x, casas_y, x0 + alpha*pk) # Ojo que phi(0) = f(x0)
    linea = lambda alpha: phi(0) + c1 * alpha * np.dot(g_x0, pk)
    g_x0 = gradiente(f, casas_x, casas_y, x0) 
    cond_1 = linea(alpha) - phi(alpha) >= 0
    cond_2 = np.dot(grad(alpha), pk) - c2 * np.dot(g_x0, pk) >= 0
    return  cond_1 and cond_2 


# Encuentro la alpha que va a cumplir las condiciones de Wolfe
def genera_alpha(f, casas_x, casas_y, x0, pk, c1 = 1e-4, tol = 1e-5):

    alpha, rho, c = 1, 4/5, c1
    while f(casas_x, casas_y, x0 + alpha * pk)>f(casas_x, casas_y, x0) + c * alpha * np.dot(gradiente(f, casas_x, casas_y, x0), pk):
        alpha*=rho
    return alpha


# Implementamos el método de máximo descenso para encontrar la mejor red de distribución de camiones
def max_des(f, casas_x, casas_y, x0):

    xk = x0

    while not (f_o_c(f, casas_x, casas_y, xk) and s_o_c(f, casas_x, casas_y, xk)):
        grad = gradiente(f, casas_x, casas_y, xk)
        pk = - grad
        alpha = genera_alpha(f, casas_x, casas_y, xk, pk)
        xk = xk + alpha * pk
    
    return xk


# Implementamos el método de Newton para encontrar la mejor red de distribución de camiones
def newton(f, casas_x, casas_y, x0):

    xk = x0

    while not (f_o_c(f, casas_x, casas_y, xk) and s_o_c(f, casas_x, casas_y, xk)):
        grad = gradiente(f, casas_x, casas_y, xk)
        hess = Hess(f, casas_x, casas_y, xk)
        pk = np.linalg.solve(hess, - grad)
        xk = xk + pk
    
    return xk


print(max_des(distancia_total, ca_x, ca_y, camiones))

#print(newton(distancia_total, ca_x, ca_y, camiones))

#print(Hess(distancia_total, ca_x, ca_y, camiones))

#np.linalg.solve(Hess(distancia_total, ca_x, ca_y, camiones), - gradiente(distancia_total, ca_x, ca_y, camiones))

tiempo_trasncurrido = time.time() - start_time
print(f"El código tardó {tiempo_trasncurrido}")

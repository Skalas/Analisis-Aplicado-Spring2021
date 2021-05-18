import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import time

start_time = time.time()

num_casas = 100
num_camiones = 5

ca_x = np.random.uniform(0, 100, num_casas)
ca_y = np.random.uniform(0, 178, num_casas)


# Vector inicial con distribución uniforme del número de camiones que queremos
min_lat= np.amin(ca_x)
max_lat = np.amax(ca_x)
min_long = np.amin(ca_y)
max_long = np.amax(ca_y)
c_x = np.random.uniform(low = min_lat, high = max_lat, size = num_camiones) # latitud (eje x) de los camiones
c_y = np.random.uniform(low = min_long, high = max_long, size = num_camiones) # longitud (eje y) de los camiones
coordenadas_casas = np.zeros(2*num_casas)
coordenadas_camiones = np.zeros(2*num_camiones)

# Aplana los datos
for i in range(num_casas):
    coordenadas_casas[i] = ca_x[i]
    coordenadas_casas[i + num_casas] = ca_y[i]
for i in range(num_camiones):
    coordenadas_camiones[i] = c_x[i]
    coordenadas_camiones[i + num_camiones] = c_y[i]

# Checo la distancia de cada casa al camión más cercano
def encuentra_minimos():
    minim = np.zeros(num_casas)
    for i in range(num_casas):
        aux = 10000000
        for j in range(num_camiones):
            dist = np.sqrt((ca_x[i] - coordenadas_camiones[j])**2 + (ca_y[i] - coordenadas_camiones[j + num_camiones])**2)
            if dist < aux:
                aux = dist
                minim[i] = j
    return minim
minimos = encuentra_minimos()
print(minimos)



def distancia_total(camiones):

    total = 0

    for i in range(num_casas):
        total = total + np.sqrt((coordenadas_casas[i] - camiones[int(minimos[i])])**2 + (coordenadas_casas[i + num_casas] - camiones[int(minimos[i]) + num_camiones])**2)

    return total

def gradiente(f, x, h = .0001):
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        z = np.zeros(n)
        z[i] = h/2
        grad[i] = (f(x + z) - f(x - z))/h
    return grad

def hessiana(f, x, h = .0001):
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        w = np.zeros(n)
        w[i] = h
        for j in range(n):
            if i==j:
                hess[i][j] = (- f(x + 2 * w) + 16 * f(x + w) - 30 * f(x) + 16 * f(x - w) -f(x - 2 * w))/(12 * h**2)
            else:
                z = np.zeros(n)
                z[j] = h
                hess[i][j] = (f(x + w + z) - f(x - w + z) - f(x - z + w) + f(x - z - w))/(4 * h**2)
    return hess

def condicionesnecesarias(f, x, h = .0001):
    resp = True
    n = len(x)
    grad = gradiente(f, x, h)
    for i in range(n):
        if abs(grad[i]) > h:
            resp = False
            break
    if resp:
        eigenValores = np.linalg.eig(hessiana(f, x, h))
        for i in range(n):
            if eigenValores[0][i] < h:
                resp = False
                break

    return resp        
####### direcciones de descenso ########
def steepest(f, x):
    return - gradiente(f, x)
def newt(f, x, h=.0001):
    return - np.matmul(np.linalg.inv(hessiana(f, x, h)),gradiente(f, x, h))

def condicionesWolfe(f, x, p, a, c1 = .5, c2 = .9):
    resp = True
    if f(x + a * p) > f(x) + c1 * a * np.dot(gradiente(f, x), p):
        resp=False
    if np.dot(gradiente(f, x + a*p), p) < c2 * np.dot(gradiente(f, x), p):
        resp=False
    return resp

    # Encuentro la alpha que va a cumplir las condiciones de Wolfe
def genera_alpha(f, x, p, c1 = 1e-4, tol = 1e-5):

    a, rho, c = 1, 4/5, c1
    while f(x + a * p)>f(x) + c1 * a * np.dot(gradiente(f, x), p):
        a*=rho
    return a

def encuentraMinimo(f,x, phi = .85, h =.0001):
    xs = []
    xs = xs + [x]
    print(xs)

    while not condicionesnecesarias(f, x, h):
        a = 1
        p = newt(f, x, h)

        while not condicionesWolfe(f, x, p, a):
            a = phi * a
        x = x + a * p
        xs = xs + [x]

    return xs


# Implementamos el método de máximo descenso para encontrar la mejor red de distribución de camiones
def max_des(f, x0, h = .0001):

    xk = x0
    xs = []
    xs = xs + [x0]
    while not condicionesnecesarias(f, xk, h):
        
        grad = gradiente(f, xk)
        pk = - grad
        alpha = genera_alpha(f, xk, pk)
        xk = xk + alpha * pk
        xs = xs + [xk]
        print(xs)
    
    return xs



fig = plt.figure()
# encuentra las coordenadas ideales con maximo descenso y con la direccion de newton
coor_final_2 = max_des(distancia_total, coordenadas_camiones)
coor_final = encuentraMinimo(distancia_total, coordenadas_camiones)
#mide el tiempo que se tardo en encontrarlas coordenadas con los dos metodos
tiempo = - start_time + time.time()
print("El código tardó %3.2f segundos" % (tiempo))
print("Pasos con direccion de newton: %3d" % len(coor_final))
print("Pasos con direccion de maximo descenso: %3d" % len(coor_final_2))

# genera la animacion
fig = plt.figure()

if len(coor_final) < len(coor_final_2):
    n = len(coor_final)
    for i in range(n, len(coor_final_2)):
        coor_final = coor_final + [coor_final[n-1]]
else:
    n = len(coor_final_2)
    for i in range(n, len(coor_final)):
        coor_final_2 = coor_final_2 + [coor_final_2[n-1]]


def animacion(i):
    xf = np.zeros(num_camiones)
    yf = np.zeros(num_camiones)
    xf_2 = np.zeros(num_camiones)
    yf_2 = np.zeros(num_camiones)
    for j in range(num_camiones):
        xf[j] = coor_final[i][j]
        yf[j] = coor_final[i][j + num_camiones]
        xf_2[j] = coor_final_2[i][j]
        yf_2[j] = coor_final_2[i][j + num_camiones]
        

    plt.plot(ca_x, ca_y, 'o', color = 'black')
    plt.plot(c_x, c_y, 'o', color = 'blue')
    plt.plot(xf, yf, 'o', color = 'red')
    plt.plot(xf_2, yf_2, 'x', color = 'green')



animator = ani.FuncAnimation(fig, animacion, init_func = plt.clf, frames = len(coor_final), interval = 500)
plt.show()

# Genera la grafica final
xf = np.zeros(num_camiones)
yf = np.zeros(num_camiones)
xf_2 = np.zeros(num_camiones)
yf_2 = np.zeros(num_camiones)

for i in range(num_camiones):
    xf[i] = coor_final[len(coor_final)-1][i]
    yf[i] = coor_final[len(coor_final)-1][i + num_camiones]

for i in range(num_camiones):
    xf_2[i] = coor_final_2[len(coor_final_2)-1][i]
    yf_2[i] = coor_final_2[len(coor_final_2)-1][i + num_camiones]


plt.plot(ca_x, ca_y, 'o', color = 'black')
plt.plot(c_x, c_y, 'o', color = 'blue')
plt.plot(xf, yf, 'o', color = 'red')
plt.plot(xf_2, yf_2, 'x', color = 'green')
plt.show()
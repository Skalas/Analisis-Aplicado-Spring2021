import numpy as np
import matplotlib.pyplot as plt

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

#aplana los datos
for i in range(num_casas):
    coordenadas_casas[i] = ca_x[i]
    coordenadas_casas[i+ num_casas] = ca_y[i]
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
print(type(minimos))



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
        grad[i] = (f(x+z) - f(x-z))/h
    return grad

def hessiana(f, x, h = .0001):
    n = len(x)
    hess = np.zeros((n,n))
    for i in range(n):
        w = np.zeros(n)
        w[i] = h
        for j in range(n):
            if i==j:
                hess[i][j] = (-f(x+2*w) +16*f(x+w) - 30*f(x) + 16*f(x-w) -f(x-2*w))/(12*h**2)
            else:
                z = np.zeros(n)
                z[j] = h
                hess[i][j] = (f(x + w + z) - f(x - w + z) - f(x - z + w) + f(x - z - w))/(4*h**2)
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
    return -gradiente(f,x)
def newt(f, x, h=.0001):
    return -np.matmul(np.linalg.inv(hessiana(f,x, h)),gradiente(f,x,h))
    #return -np.linalg.solve(hessiana(f, x, h), gradiente(f, x, h))

def condicionesWolfe(f, x, p, a, c1=.5, c2=.9):
    resp = True
    if f(x + a*p) > f(x) + c1*a*np.dot(gradiente(f, x), p):
        print(1)
        resp=False
    if np.dot(gradiente(f, x + a*p), p) < c2*np.dot(gradiente(f, x), p):
        print(2)
        resp=False
    return resp

def encuentraMinimo(f,x, phi = .85, h =.0001):
    while not condicionesnecesarias(f, x, h):
        a = 1
        #p = steepest(f, x)
        p = newt(f, x, h)
        print(x)

        while not condicionesWolfe(f, x, p, a):
            a = phi*a
            print("no cumple wolfe")
        x = x + a*p
    return x

print(coordenadas_camiones)
coor_final = encuentraMinimo(distancia_total, coordenadas_camiones)
xf = np.zeros(num_camiones)
yf = np.zeros(num_camiones)

for i in range(num_camiones):
    xf[i] = coor_final[i]
    yf[i] = coor_final[i + num_camiones]

plt.plot(ca_x, ca_y, 'o', color='black')
plt.plot(c_x, c_y, 'o', color='blue')
plt.plot(xf, yf, 'o', color='red')
plt.show()
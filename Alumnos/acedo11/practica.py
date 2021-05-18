import numpy as np
import pandas as pd

# pd.read_csv("file_name.csv")


#tomando a=1, b=100
rosenbrockfunction = lambda x,y: ((1-x)**2) + 100*(y-x**2)**2

#numero de puntos
n = 100 
m = 100 

#intervalo en x
a= -2 
b = 2 
#intervalo en y
c = -2 
d = 2 

X,Y = np.meshgrid(np.linspace(a,b,n), np.linspace(c,d,m))

Z = rosenbrockfunction(X,Y)

plt.contour(X,Y,Z,np.logspace(-0.5,3.5,20,base=10),cmap='gray')
plt.xlabel('x')
plt.ylabel('y')

plt.show()


dat = pd.read_csv("dat.csv")
dat.head()  

periodo = dat['Period']
periodo.head()
minimo = periodo.min()
print(minimo)
arr = np.array([[1,2,3], [4, 5, 6], [7, 8, 9]])
arr2 = np.ones(8)
print(arr2)

dat = pd.read_csv("dat.csv")
dat.head()  

#Maximo y minimo de longitud
long_min = dat['long'].min()
long_max = dat['long'].max()

#Maximo y minimo de latitud
lat_max = dat['lat'].min()
lat_min = dat['lat'].max()

base = long_max - long_min
altura = lat_max - lat_min

# Dividimos CDMX en 8,000 zonas, un rectángulo de 100x80
step_base = base/100
step_altura = altura/80

#creamos los arreglos
arr_base = np.ones(100)
arr_altura = np.ones(80)


# creamos nuestro "grid"
arr_base = arr_base*dat['long'].min()
arr_altura = arr_altura*dat['lat'].min()
for i in range(99):
    arr_base[i] += step_base*i
arr_base[99] = dat['long'].max()
for i in range(79):
    arr_altura[i] += step_altura*i
arr_altura[79] = dat['lat'].max()

len(arr_base)

# Creamos arreglo de

m1 = []
m2 = []

for i in range(80):
    mat1.append([])
    mat2.append([])
    for j in range(100):
        mat1[i].append([])
        mat2[i].append([])

def latIndex(lat):
    for i in range(80):
        if(lat <= arrLat[i]):
            return i

def longIndex(long):
    for i in range(100):
        if(long <= arrLong[i]):
            return i

#Ahora vamos a separar cada crimen en la zona que le toca

n = dat.shape[0]

for i in range(n):
    lat = dat['lat'][i]
    long = dat['long'][i]
    latIn = latIndex(lat)
    longIn = longIndex(long)
    m1[latIn][longIn].append(lat)
    m2[latIn][longIn].append(long)


for i in range(80):
    for j in range(100):
        if(len(m1[i][j]) != len(m2[i][j])):
            print("error")

def creaFuncion(row, col):
    n = len(m1[row][col])
    def f(x):
        #x pertenece a R2
        res = 0
        for i in range (n):
           res +=  np.sqrt((x[0] - m1[row][col][i])**2 + (x[1] - m2[row][col][i])**2)
        return res
    return f

# funciones




#respuesta
ans1 = []
ans2 = []
for i in range(80):
    for j in range(100):
        x = np.zeros(2)
        if len(mat1[i][j]) == 0:
            if i == 0:
                x[0] = (arrLat[0] + latMin)/2
            else:
                x[0] = (arrLat[i] + arrLat[i-1])/2
            if j == 0:
                x[1] = (arrLong[0] + longMin)/2
            else:
                x[1] = (arrLong[j] + arrLong[j-1])/2
        else:
            inicial = np.zeros(2)
            f = creaFuncion(i,j)
            x = line_search(f,inicial)
        ans1.append(x[0])
        ans2.append(x[1])



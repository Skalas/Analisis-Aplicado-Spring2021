import math

x = [1, 2, 3, 4, 5, 6]
t = 5


def modelo(x, t):
    y = x[0] + x[3]*math.exp(-x[1]*(x[2]-t)**2 / 4) + x[4]*math.cos(x[5]*t)
    return y


print(modelo(x, t))

y1 = [5, 4, 3, 2, 1, 9, 8, 7, 6, 5]
t1 = [10, 52, 98, 47, 89, 36, 25, 14, 78, 59]


def Costo(x, y=y1, t=t1):
    for i in range(10):
        print(i)


Costo(x)

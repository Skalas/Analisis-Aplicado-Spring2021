import numpy

def rosenbrock(X):
    x = X[0]
    y = X[1]
    a = 1. - x
    b = y - x*x
    return a*a + b*b*100.

def cost_function_rosenbrock(X, y):
    suma = 0
    x = rosenbrock(X)
    for i in x:
        print(i["x"], i["y"])
        suma += (i["x"] - y["x"])**2 + (i["y"] - y["y"])**2
    return suma

